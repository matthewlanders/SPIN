import math
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Any, Dict, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from dmc_datasets.environment_utils import make_env


@dataclass
class ArgsASM:
    exp_name: str = "ASM_pretrain"
    seed: int = 1
    torch_deterministic: bool = False
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "interact"
    wandb_entity: Optional[str] = None
    runs_dir: str = None
    save_model: bool = True
    model_save_dir: str = None
    save_optimizer_state: bool = False

    domain_name: str = "dog"
    task_name: str = "trot"
    dataset_level: str = "medium-expert"
    dataset_root_dir: str = 'datasets'
    bin_size: int = 3

    d_model: int = 256
    num_heads: int = 4
    num_transformer_blocks: int = 3
    num_state_embeddings_M: int = 1
    use_state_in_ASM_core: bool = True
    prediction_head_hidden_size: int = 128
    transformer_dropout_p: float = 0.1
    ffn_hidden_mult: int = 2

    batch_size: int = 1024
    learning_rate: float = 4e-4
    num_epochs: int = 100

    lr_schedule: str = "constant"
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0
    min_lr: float = 0.0

    masking_prob: float = 0.15
    bert_mask_proportion: float = 0.8
    bert_random_proportion: float = 0.1

    grad_norm_clip: float = 1.0
    loss_norm_eps: float = 1e-8

    dataloader_num_workers: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: Optional[int] = 2

    state_dim: int = field(init=False)
    action_cardinalities: Tuple[int, ...] = field(default_factory=tuple)
    num_sub_actions_N: int = field(init=False)
    checkpoint_interval: int = 10
    mask_token_ids: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.model_save_dir is None:
            self.model_save_dir = "ASM"

        if self.bin_size == 3:
            self.save_model_path_dir = os.path.join(
                self.model_save_dir,
                self.dataset_level,
                self.domain_name,
                str(self.seed)
            )
        else:
            self.save_model_path_dir = os.path.join(
                self.model_save_dir,
                self.dataset_level,
                self.domain_name,
                str(self.bin_size),
                str(self.seed)
            )

        self.save_model_path = os.path.join(self.save_model_path_dir, "final.pt")

        if not hasattr(self, 'runs_dir') or not self.runs_dir:
            if self.bin_size == 3:
                self.runs_dir = os.path.join(os.getcwd(), "runs",  self.dataset_level, self.domain_name)
            else:
                self.runs_dir = os.path.join(os.getcwd(), "runs", self.dataset_level, self.domain_name,
                                             str(self.bin_size))


class ASMReplayBuffer:
    def __init__(self, device: torch.device, env_name: str, task_name: str,
                 dataset_level: str, dataset_root_dir: str, bin_size: int):
        self.device = device
        print(f"Setting up environment and loading ASM data for: {env_name}-{task_name}")
        try:
            env = make_env(task_name=env_name, task=task_name, factorised=True, bin_size=bin_size)
            raw_data_tuples = env.load_dataset(level=dataset_level, data_dir=dataset_root_dir)
            env.close()

            if not raw_data_tuples:
                raise ValueError("Loaded dataset is empty.")

            observations = np.array([t[0] for t in raw_data_tuples], dtype=np.float32)
            actions = np.array([t[1] for t in raw_data_tuples], dtype=np.int64)

            self.states = torch.from_numpy(observations).to(self.device)
            self.actions = torch.from_numpy(actions).to(self.device)

            self.size = self.states.shape[0]
            print(f"ASM dataset loaded to device '{self.device}'. Size: {self.size}")

        except Exception as e:
            print(f"Fatal: Error loading ASM dataset: {e}")
            raise e

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.states[indices], self.actions[indices]


class MABMod(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ffn_hidden_mult=2, dropout_p=0.1):
        super().__init__()
        self.num_heads = num_heads
        assert dim_V % num_heads == 0
        self.dim_v_per_head = dim_V // num_heads

        self.ln_q = nn.LayerNorm(dim_Q)
        self.ln_kv = nn.LayerNorm(dim_K)

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        self.attn_dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Linear(dim_V, dim_Q)
        self.out_dropout = nn.Dropout(dropout_p)

        self.ffn_ln = nn.LayerNorm(dim_Q)
        self.ffn = nn.Sequential(
            nn.Linear(dim_Q, dim_Q * ffn_hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim_Q * ffn_hidden_mult, dim_Q),
            nn.Dropout(dropout_p)
        )

    def forward(self, Q, K, V_in=None, attention_mask=None):
        B, N_q, D_q = Q.shape
        _, N_k, _ = K.shape

        Q_norm = self.ln_q(Q)
        K_norm = self.ln_kv(K)
        V_norm_in = self.ln_kv(K if V_in is None else V_in)

        q_proj = self.fc_q(Q_norm).view(B, N_q, self.num_heads, self.dim_v_per_head).permute(0, 2, 1, 3)
        k_proj = self.fc_k(K_norm).view(B, N_k, self.num_heads, self.dim_v_per_head).permute(0, 2, 3, 1)
        v_proj = self.fc_v(V_norm_in).view(B, N_k, self.num_heads, self.dim_v_per_head).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q_proj, k_proj) / (self.dim_v_per_head ** 0.5)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output_multihead = torch.matmul(attn_probs, v_proj)
        attn_output_concat = attn_output_multihead.permute(0, 2, 1, 3).contiguous().view(B, N_q,
                                                                                         self.num_heads * self.dim_v_per_head)

        x = Q + self.out_dropout(self.out_proj(attn_output_concat))
        x_ffn_norm = self.ffn_ln(x)
        ffn_out = self.ffn(x_ffn_norm)
        x = x + ffn_out
        return x, attn_probs


class SABMod(nn.Module):
    def __init__(self, dim_in_out, num_heads, ffn_hidden_mult=2, dropout_p=0.1):
        super().__init__()
        self.mab = MABMod(dim_in_out, dim_in_out, dim_in_out, num_heads, ffn_hidden_mult, dropout_p)

    def forward(self, X, attention_mask=None):
        return self.mab(X, X, V_in=X, attention_mask=attention_mask)


class ASMCore(nn.Module):
    def __init__(self, state_dim: int, num_sub_actions: int, d_model: int,
                 num_heads: int, num_tf_blocks: int, action_cardinalities: Tuple[int, ...],
                 num_state_embeddings_M: int = 1, use_state_encoder: bool = True,
                 dropout_p: float = 0.1, ffn_hidden_mult: int = 2):
        super().__init__()
        self.num_sub_actions_N = num_sub_actions
        self.d_model = d_model
        self.num_state_embeddings_M = num_state_embeddings_M if use_state_encoder else 0
        self.use_state_encoder = use_state_encoder
        self.action_cardinalities = action_cardinalities

        if self.use_state_encoder:
            assert num_state_embeddings_M > 0, "M must be > 0 if use_state_encoder is True"
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, d_model * 2), nn.GELU(),
                nn.Linear(d_model * 2, self.num_state_embeddings_M * d_model)
            )
        else:
            self.state_encoder = None

        self.sub_action_value_embeddings = nn.ModuleList()
        for card in action_cardinalities:
            self.sub_action_value_embeddings.append(nn.Embedding(card + 1, d_model))

        self.transformer_blocks = nn.ModuleList(
            [SABMod(d_model, num_heads, ffn_hidden_mult=ffn_hidden_mult, dropout_p=dropout_p) for _ in range(num_tf_blocks)]
        )

    def forward(self, state_features: Optional[torch.Tensor],
                input_action_ids_batch: torch.Tensor):

        if self.use_state_encoder and state_features is None:
            raise ValueError("state_features must be provided when use_state_encoder is True")

        if state_features is not None:
            B = state_features.shape[0]
        else:
            B = input_action_ids_batch.shape[0]

        embedded_actions_list = [
            self.sub_action_value_embeddings[i](input_action_ids_batch[:, i])
            for i in range(self.num_sub_actions_N)
        ]
        batch_sub_action_token_embeddings = torch.stack(embedded_actions_list, dim=1)

        if self.use_state_encoder and state_features is not None and self.state_encoder is not None:
            state_token_embeddings = self.state_encoder(state_features).view(B, self.num_state_embeddings_M,
                                                                             self.d_model)
            transformer_input = torch.cat([state_token_embeddings, batch_sub_action_token_embeddings], dim=1)
        else:
            transformer_input = batch_sub_action_token_embeddings

        current_x = transformer_input
        all_layer_attentions = []
        for tf_block in self.transformer_blocks:
            current_x, layer_attention_maps = tf_block(current_x)
            all_layer_attentions.append(layer_attention_maps)

        contextual_slot_embeddings = current_x[:, self.num_state_embeddings_M:, :]
        return contextual_slot_embeddings, all_layer_attentions


class ActionStructureModel(nn.Module):
    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...],
                 d_model: int, num_heads: int, num_tf_blocks: int,
                 num_state_embeddings_M: int = 1, use_state_in_core: bool = True,
                 prediction_head_hidden_size: int = 128, dropout_p: float = 0.1,
                 ffn_hidden_mult_core: int = 2):
        super().__init__()
        self.num_sub_actions_N = len(action_cardinalities)

        self.core_transformer = ASMCore(
            state_dim=state_dim, num_sub_actions=self.num_sub_actions_N,
            d_model=d_model, num_heads=num_heads, num_tf_blocks=num_tf_blocks,
            action_cardinalities=action_cardinalities,
            num_state_embeddings_M=num_state_embeddings_M,
            use_state_encoder=use_state_in_core, dropout_p=dropout_p,
            ffn_hidden_mult=ffn_hidden_mult_core
        )
        self.prediction_heads = nn.ModuleList()
        for card in action_cardinalities:
            self.prediction_heads.append(nn.Sequential(
                nn.Linear(d_model, prediction_head_hidden_size), nn.GELU(),
                nn.Dropout(dropout_p), nn.Linear(prediction_head_hidden_size, card)
            ))

    def forward(self, state_features: Optional[torch.Tensor],
                input_action_ids_for_core: torch.Tensor):

        contextual_slot_embeddings, _ = self.core_transformer(
            state_features,
            input_action_ids_batch=input_action_ids_for_core
        )
        output_logits_list = []
        for i in range(self.num_sub_actions_N):
            logits = self.prediction_heads[i](contextual_slot_embeddings[:, i, :])
            output_logits_list.append(logits)
        return output_logits_list

    @torch.no_grad()
    def get_contextual_slot_embeddings_and_attentions(self,
                                                      state_features: Optional[torch.Tensor],
                                                      observed_action_ids_batch: torch.Tensor):
        self.core_transformer.eval()
        slot_embeds, attention_maps_list = self.core_transformer(
            state_features,
            input_action_ids_batch=observed_action_ids_batch
        )
        if attention_maps_list:
            stacked_attention_maps = torch.stack(attention_maps_list, dim=0)
            return slot_embeds, stacked_attention_maps
        return slot_embeds, []


def train_ASM(args: ArgsASM, writer: Optional[SummaryWriter], device: torch.device = torch.device("cpu")):
    try:
        replay_buffer = ASMReplayBuffer(
            device=device,
            env_name=args.domain_name,
            task_name=args.task_name,
            dataset_level=args.dataset_level,
            dataset_root_dir=args.dataset_root_dir,
            bin_size=args.bin_size,
        )
    except Exception as e:
        print(f"Failed to create ASMReplayBuffer: {e}")
        return

    num_batches_per_epoch = (replay_buffer.size + args.batch_size - 1) // args.batch_size

    ASM_network = ActionStructureModel(
        state_dim=args.state_dim, action_cardinalities=args.action_cardinalities,
        d_model=args.d_model, num_heads=args.num_heads,
        num_tf_blocks=args.num_transformer_blocks,
        num_state_embeddings_M=args.num_state_embeddings_M,
        use_state_in_core=args.use_state_in_ASM_core,
        prediction_head_hidden_size=args.prediction_head_hidden_size,
        dropout_p=args.transformer_dropout_p,
        ffn_hidden_mult_core=args.ffn_hidden_mult
    ).to(device)

    if args.cuda and hasattr(torch, "compile"):
        try:
            ASM_network = torch.compile(ASM_network)
        except Exception:
            pass

    optimizer = optim.Adam(ASM_network.parameters(), lr=args.learning_rate)
    amp_enabled = False
    amp_dtype = torch.float32
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    IGNORE_IDX = -100

    total_training_steps = max(1, num_batches_per_epoch * args.num_epochs)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(args.warmup_ratio * total_training_steps)
    warmup_steps = max(0, min(warmup_steps, total_training_steps - 1))

    base_lr = args.learning_rate
    final_factor = (args.min_lr / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if args.lr_schedule == "cosine":
            if total_training_steps == warmup_steps:
                return final_factor
            progress = (current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_factor + (1.0 - final_factor) * cosine
        else:
            return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    device_mask_token_ids = torch.tensor(args.mask_token_ids, device=device, dtype=torch.long) if args.mask_token_ids else torch.empty(0, device=device, dtype=torch.long)
    device_action_cardinalities = torch.tensor(args.action_cardinalities, device=device, dtype=torch.long) if args.action_cardinalities else torch.empty(0, device=device, dtype=torch.long)

    print(f"Starting ASM pre-training for {args.num_epochs} epochs (80/10/10 MAM)...")
    global_step = 0
    epoch = 0
    for epoch in range(args.num_epochs):
        ASM_network.train()
        total_epoch_loss = 0
        total_tokens_for_loss_in_epoch = 0

        for batch_idx in range(num_batches_per_epoch):
            states_batch, true_actions_batch = replay_buffer.sample(args.batch_size)
            optimizer.zero_grad(set_to_none=True)
            current_state_features = states_batch if args.use_state_in_ASM_core else None

            B, N_sub_actions = true_actions_batch.shape

            if N_sub_actions == 0: continue

            input_action_ids_for_core = true_actions_batch.clone()
            loss_calculation_mask = torch.zeros_like(true_actions_batch, dtype=torch.bool, device=device)

            bern_mask = torch.rand_like(true_actions_batch, dtype=torch.float) < args.masking_prob
            if B > 0 and N_sub_actions > 0 and not bern_mask.any():
                rand_batch_idx = random.randint(0, B - 1)
                rand_action_idx = random.randint(0, N_sub_actions - 1)
                bern_mask[rand_batch_idx, rand_action_idx] = True

            loss_calculation_mask[bern_mask] = True

            manipulate_rows, manipulate_cols = torch.where(bern_mask)

            if manipulate_rows.numel() > 0:
                num_manipulated = manipulate_rows.size(0)
                fate = torch.rand(num_manipulated, device=device)

                mask_input_condition = fate < args.bert_mask_proportion
                if mask_input_condition.any():
                    m_rows = manipulate_rows[mask_input_condition]
                    m_cols = manipulate_cols[mask_input_condition]
                    if device_mask_token_ids.numel() > 0 and m_cols.max() < device_mask_token_ids.size(0):
                         input_action_ids_for_core[m_rows, m_cols] = device_mask_token_ids[m_cols]

                random_input_condition = (fate >= args.bert_mask_proportion) & \
                                       (fate < args.bert_mask_proportion + args.bert_random_proportion)
                if random_input_condition.any():
                    r_rows = manipulate_rows[random_input_condition]
                    r_cols = manipulate_cols[random_input_condition]
                    if device_action_cardinalities.numel() > 0 and r_cols.max() < device_action_cardinalities.size(0):
                        cards_for_r_cols = device_action_cardinalities[r_cols]
                        valid_cards = torch.max(cards_for_r_cols, torch.ones_like(cards_for_r_cols))
                        random_action_values = (torch.rand(r_rows.size(0), device=device) * valid_cards).long()
                        random_action_values = torch.min(random_action_values, torch.max(torch.zeros_like(valid_cards), valid_cards - 1))
                        input_action_ids_for_core[r_rows, r_cols] = random_action_values

            targets_for_loss = true_actions_batch.clone()
            targets_for_loss[~loss_calculation_mask] = IGNORE_IDX

            with torch.amp.autocast(enabled=amp_enabled, dtype=amp_dtype, device_type=device.type):
                predicted_logits_list = ASM_network(
                    current_state_features,
                    input_action_ids_for_core
                )

                current_batch_loss_tensor = torch.tensor(0.0, device=device)
                num_tokens_for_loss_in_batch = 0
                for i in range(args.num_sub_actions_N):
                    loss_sub_action = F.cross_entropy(
                        predicted_logits_list[i], targets_for_loss[:, i],
                        ignore_index=IGNORE_IDX, reduction='sum'
                    )
                    current_batch_loss_tensor += loss_sub_action
                    num_tokens_for_loss_in_batch += (targets_for_loss[:, i] != IGNORE_IDX).sum().item()

            if num_tokens_for_loss_in_batch > 0:
                mean_batch_loss_tensor = current_batch_loss_tensor / (num_tokens_for_loss_in_batch + args.loss_norm_eps)
                scaler.scale(mean_batch_loss_tensor).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ASM_network.parameters(), args.grad_norm_clip)

                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= old_scale
                if did_step:
                    scheduler.step()

                total_epoch_loss += current_batch_loss_tensor.item()
                total_tokens_for_loss_in_epoch += num_tokens_for_loss_in_batch
                if writer and global_step % 100 == 0:
                    writer.add_scalar("ASM_train/lr", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("ASM_train/batch_mam_loss", mean_batch_loss_tensor.item(), global_step)
            global_step += 1

        avg_epoch_loss = total_epoch_loss / (total_tokens_for_loss_in_epoch + args.loss_norm_eps) if total_tokens_for_loss_in_epoch > 0 else 0.0
        print(f"Epoch {epoch}/{args.num_epochs}, Avg MAM Loss: {avg_epoch_loss:.4f}")
        if writer:
            writer.add_scalar("ASM_train/epoch_mam_loss", avg_epoch_loss, epoch)

        if args.save_model and epoch % args.checkpoint_interval == 0:
            os.makedirs(args.save_model_path_dir, exist_ok=True)
            ckpt_path = os.path.join(
                args.save_model_path_dir,
                f"{epoch}.pt"
            )
            model_to_save = ASM_network._orig_mod if hasattr(ASM_network, '_orig_mod') else ASM_network
            checkpoint = {
                'epoch': epoch,
                'args': asdict(args),
                'model_state_dict': model_to_save.state_dict(),
            }
            if args.save_optimizer_state:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        model_to_save = ASM_network._orig_mod if hasattr(ASM_network, '_orig_mod') else ASM_network
        checkpoint = {'args': asdict(args), 'model_state_dict': model_to_save.state_dict()}
        if args.save_optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, args.save_model_path)
        print(f"ASM model and args saved to {args.save_model_path}")


if __name__ == "__main__":
    args = tyro.cli(ArgsASM)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if device.type == "cuda" and args.torch_deterministic:
         torch.cuda.manual_seed_all(args.seed)

    try:
        temp_env_ASM = make_env(task_name=args.domain_name, task=args.task_name, factorised=True,
                                bin_size=args.bin_size)
        args.state_dim = temp_env_ASM.observation_space.shape[0]
        if not hasattr(temp_env_ASM.action_space, 'nvec'):
            raise ValueError(f"Action space for {args.domain_name}-{args.task_name} lacks 'nvec'.")
        args.action_cardinalities = tuple(temp_env_ASM.action_space.nvec)
        args.num_sub_actions_N = len(args.action_cardinalities)
        args.mask_token_ids = [card for card in args.action_cardinalities]
    except Exception as e:
        print(f"Error determining environment specifications: {e}")
        exit()

    run_name = f"ASM-{args.domain_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=vars(args), name=run_name, save_code=True)
    writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    train_ASM(args, writer, device)

    if writer:
        writer.close()
    if args.track and 'wandb' in locals() and wandb.run is not None:
        wandb.finish()
