import os
import random
import time
import math
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import tyro


class ActorMLP(nn.Module):
    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...],
                 hidden_dim: int = 256, num_hidden_layers: int = 2,
                 dropout_p: float = 0.0, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.N = len(action_cardinalities)
        self.cards = action_cardinalities

        layers, d = [], state_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            if dropout_p > 0: layers.append(nn.Dropout(dropout_p))
            d = hidden_dim
        self.shared = nn.Sequential(*layers).to(device)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, c).to(device) for c in self.cards])

    def forward(self, states: torch.Tensor) -> List[torch.Tensor]:
        h = self.shared(states)
        return [F.log_softmax(head(h), dim=-1) for head in self.heads]

    @torch.no_grad()
    def sample_one(self, states: torch.Tensor) -> torch.Tensor:
        h = self.shared(states)
        outs = []
        for head in self.heads:
            logits = head(h)
            outs.append(torch.distributions.Categorical(logits=logits).sample())
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def sample_actions_K(self, states: torch.Tensor, K: int) -> torch.Tensor:
        h = self.shared(states)
        samples_per_slot = []
        for head in self.heads:
            logits = head(h)
            dist = torch.distributions.Categorical(logits=logits)
            a_samples = dist.sample((K,)).permute(1, 0).contiguous()
            samples_per_slot.append(a_samples)
        return torch.stack(samples_per_slot, dim=0).permute(1, 2, 0).contiguous()


class MABMod(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ffn_hidden_mult=2, dropout_p=0.1):
        super().__init__()
        self.num_heads = num_heads
        if dim_V % num_heads != 0:
            raise ValueError(f"dim_V ({dim_V}) must be divisible by num_heads ({num_heads})")
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
            nn.Linear(dim_Q, dim_Q * ffn_hidden_mult), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(dim_Q * ffn_hidden_mult, dim_Q), nn.Dropout(dropout_p)
        )

    def forward(self, Q, K, V_in=None, attention_mask=None):
        B, N_q, _ = Q.shape
        _, N_k, _ = K.shape
        Q_norm = self.ln_q(Q)
        K_norm = self.ln_kv(K)
        V_norm_in = self.ln_kv(K if V_in is None else V_in)
        q_proj = self.fc_q(Q_norm).view(B, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k_proj = self.fc_k(K_norm).view(B, N_k, self.num_heads, -1).permute(0, 2, 3, 1)
        v_proj = self.fc_v(V_norm_in).view(B, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q_proj, k_proj) / (q_proj.size(-1) ** 0.5)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v_proj)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N_q, -1)
        x = Q + self.out_dropout(self.out_proj(attn_output))
        x = x + self.ffn(self.ffn_ln(x))
        return x, attn_probs

class SABMod(nn.Module):
    def __init__(self, dim_in_out, num_heads, ffn_hidden_mult=2, dropout_p=0.1):
        super().__init__()
        self.mab = MABMod(dim_in_out, dim_in_out, dim_in_out, num_heads, ffn_hidden_mult, dropout_p)

    def forward(self, X, attention_mask=None):
        return self.mab(X, X, V_in=X, attention_mask=attention_mask)


class ActorSAINT(nn.Module):
    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...],
                 actor_d_model: int = 256, actor_num_heads: int = 4,
                 actor_num_tf_blocks: int = 3, actor_num_state_embeddings_M: int = 1,
                 actor_slot_emb_dropout_p: float = 0.0, actor_transformer_dropout_p: float = 0.1,
                 actor_decision_head_hidden_size: int = 128, actor_dropout_p: float = 0.1,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.N = len(action_cardinalities)
        self.cards = action_cardinalities
        self.d_model = actor_d_model
        self.M = actor_num_state_embeddings_M
        self.device = device

        self.state_encoder = None
        if self.M > 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, self.d_model * 2), nn.GELU(),
                nn.Linear(self.d_model * 2, self.M * self.d_model)
            ).to(device)

        self.slot_embeddings = nn.Parameter(torch.randn(self.N, self.d_model))
        self.slot_dropout = nn.Dropout(actor_slot_emb_dropout_p)

        self.blocks = nn.ModuleList([
            SABMod(self.d_model, actor_num_heads, 2, actor_transformer_dropout_p)
            for _ in range(actor_num_tf_blocks)
        ]).to(device)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, actor_decision_head_hidden_size), nn.GELU(),
                nn.Dropout(actor_dropout_p), nn.Linear(actor_decision_head_hidden_size, card)
            ).to(device) for card in self.cards
        ])

    def _prep_tokens(self, s: torch.Tensor) -> torch.Tensor:
        B = s.size(0)
        slots = self.slot_dropout(self.slot_embeddings.unsqueeze(0).repeat(B, 1, 1))
        if self.M > 0 and self.state_encoder is not None:
            st = self.state_encoder(s).view(B, self.M, self.d_model)
            return torch.cat([st, slots], dim=1)
        return slots

    def _pass(self, s: torch.Tensor) -> torch.Tensor:
        x = self._prep_tokens(s)
        for blk in self.blocks:
            x, _ = blk(x)
        return x[:, self.M:, :] if self.M > 0 else x

    def forward(self, states: torch.Tensor) -> List[torch.Tensor]:
        ctx = self._pass(states)
        out = []
        for i, head in enumerate(self.heads):
            logits = head(ctx[:, i, :])
            out.append(F.log_softmax(logits, dim=-1))
        return out

    @torch.no_grad()
    def sample_one(self, states: torch.Tensor) -> torch.Tensor:
        ctx = self._pass(states)
        outs = []
        for i, head in enumerate(self.heads):
            logits = head(ctx[:, i, :])
            outs.append(torch.distributions.Categorical(logits=logits).sample())
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def sample_actions_K(self, states: torch.Tensor, K: int) -> torch.Tensor:
        B = states.size(0)
        ctx = self._pass(states)
        samples = []
        for i, head in enumerate(self.heads):
            logits = head(ctx[:, i, :])
            dist = torch.distributions.Categorical(logits=logits)
            aK = dist.sample((K,)).permute(1, 0).contiguous()
            samples.append(aK)
        return torch.stack(samples, dim=0).permute(1, 2, 0).contiguous()


class ActorAR(nn.Module):
    autoregressive = True

    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...],
                 state_embed_dim: int = 128,
                 action_embed_dim: int = 32,
                 lstm_hidden_dim: int = 256,
                 lstm_layers: int = 2,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.N = len(action_cardinalities)
        self.cards = action_cardinalities
        self.device = device
        self.state_embed = nn.Sequential(nn.Linear(state_dim, state_embed_dim), nn.ReLU()).to(device)

        self.max_card = max(self.cards)
        self.start_token_idx = self.max_card
        self.action_embed = nn.Embedding(self.max_card + 1, action_embed_dim).to(device)

        self.lstm = nn.LSTM(input_size=state_embed_dim + action_embed_dim,
                            hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True).to(device)

        self.heads = nn.ModuleList([nn.Linear(lstm_hidden_dim, c).to(device) for c in self.cards])
        self.lstm_layers = lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim

    def _init_hidden(self, B: int):
        h0 = torch.zeros(self.lstm_layers, B, self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def teacher_forcing_logps(self, states: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        B = states.size(0)
        s_emb = self.state_embed(states)
        hidden = self._init_hidden(B)
        prev = self.action_embed(torch.full((B,), self.start_token_idx, dtype=torch.long, device=self.device))

        out = []
        for i in range(self.N):
            x = torch.cat([s_emb, prev], dim=1).unsqueeze(1)
            y, hidden = self.lstm(x, hidden)
            logits = self.heads[i](y.squeeze(1))
            out.append(F.log_softmax(logits, dim=-1))
            prev = self.action_embed(actions[:, i])
        return out

    @torch.no_grad()
    def sample_one(self, states: torch.Tensor) -> torch.Tensor:
        B = states.size(0)
        s_emb = self.state_embed(states)
        hidden = self._init_hidden(B)
        prev = self.action_embed(torch.full((B,), self.start_token_idx, dtype=torch.long, device=self.device))
        samples = []
        for i in range(self.N):
            x = torch.cat([s_emb, prev], dim=1).unsqueeze(1)
            y, hidden = self.lstm(x, hidden)
            logits = self.heads[i](y.squeeze(1))
            a_i = torch.distributions.Categorical(logits=logits).sample()
            samples.append(a_i)
            prev = self.action_embed(a_i)
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def sample_actions_K(self, states: torch.Tensor, K: int) -> torch.Tensor:
        B = states.size(0)
        s_emb = self.state_embed(states)
        allK = []
        for _ in range(K):
            hidden = self._init_hidden(B)
            prev = self.action_embed(torch.full((B,), self.start_token_idx, dtype=torch.long, device=self.device))
            samples = []
            for i in range(self.N):
                x = torch.cat([s_emb, prev], dim=1).unsqueeze(1)
                y, hidden = self.lstm(x, hidden)
                logits = self.heads[i](y.squeeze(1))
                a_i = torch.distributions.Categorical(logits=logits).sample()
                samples.append(a_i)
                prev = self.action_embed(a_i)
            allK.append(torch.stack(samples, dim=1))
        return torch.stack(allK, dim=1)


try:
    from dmc_datasets.environment_utils import make_env
except ImportError:
    print("Fatal: dmc_datasets.environment_utils.make_env not found. This is required.")
    raise


class FactorizedQ(nn.Module):
    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...], hidden_dim: int = 256):
        super().__init__()
        self.cards = action_cardinalities
        self.total = sum(self.cards)
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, self.total)

    def forward(self, s: torch.Tensor) -> List[torch.Tensor]:
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        q_flat = self.out(x)
        outs, idx = [], 0
        for c in self.cards:
            outs.append(q_flat[:, idx:idx+c])
            idx += c
        return outs


class TwinFactorizedCritic(nn.Module):
    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...], hidden_dim: int = 256):
        super().__init__()
        self.q1 = FactorizedQ(state_dim, action_cardinalities, hidden_dim)
        self.q2 = FactorizedQ(state_dim, action_cardinalities, hidden_dim)

    def all_q(self, s: torch.Tensor):
        return self.q1(s), self.q2(s)

    def q_sum_of(self, s: torch.Tensor, a: torch.Tensor):
        q1_list, q2_list = self.all_q(s)
        q1 = torch.zeros(s.size(0), device=s.device)
        q2 = torch.zeros_like(q1)
        for i in range(len(q1_list)):
            ai = a[:, i].unsqueeze(-1)
            q1 += torch.gather(q1_list[i], 1, ai).squeeze(-1)
            q2 += torch.gather(q2_list[i], 1, ai).squeeze(-1)
        return q1, q2


class Value(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, s):
        return self.l3(F.relu(self.l2(F.relu(self.l1(s))))).squeeze(-1)


class OfflineReplayBuffer:
    def __init__(self, device: torch.device, env_name: str, task_name: str,
                 dataset_level: str, dataset_root_dir: str):
        self.device = device
        env = make_env(task_name=env_name, task=task_name, factorised=True)
        raw = env.load_dataset(level=dataset_level, data_dir=dataset_root_dir)
        env.close()
        if not raw: raise ValueError("Loaded dataset is empty.")
        obs  = np.array([t[0] for t in raw], dtype=np.float32)
        acts = np.array([t[1] for t in raw], dtype=np.int64)
        rews = np.array([t[2] for t in raw], dtype=np.float32).reshape(-1, 1)
        nxt  = np.array([t[3] for t in raw], dtype=np.float32)
        dones= np.array([t[4] for t in raw], dtype=np.float32).reshape(-1, 1)
        self.states = torch.from_numpy(obs).to(device)
        self.actions= torch.from_numpy(acts).to(device)
        self.rewards= torch.from_numpy(rews).to(device)
        self.next_states= torch.from_numpy(nxt).to(device)
        self.dones = torch.from_numpy(dones).to(device)
        self.size  = self.states.shape[0]

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.states.device)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])


class Agent_AWAC:
    def __init__(self, *, state_dim: int, action_cardinalities: Tuple[int, ...],
                 actor_ctor,
                 actor_kwargs: dict,
                 critic_hidden_dim: int = 256, value_hidden_dim: int = 256,
                 bc_alpha: float = 0.0,
                 awac_lambda: float = 0.3, weight_clip: float = 100.0, normalize_weights: bool = False,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, lr_decay: float = 1.0,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.cards = action_cardinalities
        self.N = len(self.cards)
        self.gamma, self.tau = gamma, tau
        self.bc_alpha = bc_alpha
        self.awac_lambda = awac_lambda
        self.weight_clip = weight_clip
        self.normalize_weights = normalize_weights

        self.value_net = Value(state_dim, value_hidden_dim).to(device)
        self.value_target = copy.deepcopy(self.value_net).to(device)
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.value_sched = ExponentialLR(self.value_opt, gamma=lr_decay)

        self.actor = actor_ctor(state_dim=state_dim, action_cardinalities=action_cardinalities, **actor_kwargs).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TwinFactorizedCritic(state_dim, action_cardinalities, critic_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.actor_sched = ExponentialLR(self.actor_opt, gamma=lr_decay)
        self.critic_sched = ExponentialLR(self.critic_opt, gamma=lr_decay)

    @torch.no_grad()
    def select_action(self, state_np: np.ndarray):
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor.sample_one(s)
        return a.squeeze(0).cpu().numpy()

    def train(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, s2: torch.Tensor, d: torch.Tensor):
        log = {}
        with torch.no_grad():
            v_next = self.value_target(s2)
            y_q = r.squeeze(-1) + (1.0 - d.squeeze(-1)) * self.gamma * v_next
        q1, q2 = self.critic.q_sum_of(s, a)
        critic_loss = F.mse_loss(q1, y_q) + F.mse_loss(q2, y_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.critic_sched.step()
        log['critic_loss'] = critic_loss.item()

        with torch.no_grad():
            q1d, q2d = self.critic.q_sum_of(s, a)
            qd = torch.minimum(q1d, q2d)
            v_s = self.value_net(s)
            adv = qd - v_s
            max_log_w = math.log(self.weight_clip)
            w = torch.exp((adv / self.awac_lambda).clamp(max=max_log_w))
            if self.normalize_weights:
                w = w / (w.mean() + 1e-8)

        if getattr(self.actor, "autoregressive", False):
            logps = self.actor.teacher_forcing_logps(s, a)
        else:
            logps = self.actor(s)

        sum_logp = 0.0
        for i in range(self.N):
            sum_logp = sum_logp + torch.gather(logps[i], 1, a[:, i].unsqueeze(-1)).squeeze(-1)
        actor_loss_awac = -(w * sum_logp).mean()
        actor_loss = actor_loss_awac + self.bc_alpha * (-(sum_logp).mean())
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.actor_sched.step()
        log['actor_loss_awac'] = actor_loss_awac.item()
        log['actor_loss_total'] = actor_loss.item()

        with torch.no_grad():
            if getattr(self.actor, "autoregressive", False):
                K = 4
                aK = self.actor.sample_actions_K(s, K)
                B = s.size(0)
                s_rep = s.unsqueeze(1).repeat(1, K, 1).view(B * K, -1)
                a_rep = aK.view(B * K, self.N)
                q1k, q2k = self.critic_target.q_sum_of(s_rep, a_rep)
                v_tgt = torch.minimum(q1k, q2k).view(B, K).mean(dim=1)
            else:
                logps = self.actor(s)
                pis = [lp.exp() for lp in logps]
                q1_list, q2_list = self.critic_target.all_q(s)
                v1 = 0.0
                v2 = 0.0
                for i in range(self.N):
                    v1 = v1 + (pis[i] * q1_list[i]).sum(dim=1)
                    v2 = v2 + (pis[i] * q2_list[i]).sum(dim=1)
                v_tgt = torch.minimum(v1, v2)

        v_pred = self.value_net(s)
        v_loss = F.mse_loss(v_pred, v_tgt)
        self.value_opt.zero_grad()
        v_loss.backward()
        self.value_opt.step()
        self.value_sched.step()
        log['value_loss'] = v_loss.item()
        log['v_mean'] = v_pred.mean().item()

        # Polyak
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.value_net.parameters(), self.value_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return log


def load_asm_weights_into_actor_saint(
    actor: ActorSAINT,
    asm_ckpt_path: str,
    device: torch.device,
    freeze_loaded: bool,
    *,
    state_encoder_prefix: str = "core_transformer.state_encoder.",
    tf_blocks_prefix: str = "core_transformer.transformer_blocks."
):
    print(f"[ASM] Loading ASM checkpoint: {asm_ckpt_path}")
    if not os.path.exists(asm_ckpt_path):
        raise FileNotFoundError(f"ASM checkpoint not found: {asm_ckpt_path}")

    ckpt = torch.load(asm_ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        print("[ASM] No 'model_state_dict' key in checkpoint; skipping.")
        return
    src_sd = ckpt["model_state_dict"]

    # ---- State encoder ----
    if actor.state_encoder is not None:
        se_weights = {k.replace(state_encoder_prefix, ""): v
                      for k, v in src_sd.items() if k.startswith(state_encoder_prefix)}
        if se_weights:
            try:
                actor.state_encoder.load_state_dict(se_weights, strict=True)
                print("[ASM] Loaded state_encoder (strict=True).")
            except RuntimeError as e:
                print(f"[ASM] state_encoder strict load failed: {e} → retry strict=False")
                actor.state_encoder.load_state_dict(se_weights, strict=False)
                print("[ASM] Loaded state_encoder (strict=False).")
            if freeze_loaded:
                for p in actor.state_encoder.parameters():
                    p.requires_grad = False
                print("[ASM] Froze state_encoder.")
        else:
            print("[ASM] No weights matched state_encoder prefix; skipped.")
    else:
        print("[ASM] ActorSAINT has no state_encoder; skipped.")

    # ---- Transformer blocks (ModuleList) ----
    blk_weights = {k.replace(tf_blocks_prefix, ""): v
                   for k, v in src_sd.items() if k.startswith(tf_blocks_prefix)}
    if blk_weights:
        try:
            actor.blocks.load_state_dict(blk_weights, strict=True)
            print("[ASM] Loaded transformer blocks (strict=True).")
        except RuntimeError as e:
            print(f"[ASM] blocks strict load failed: {e} → retry strict=False")
            actor.blocks.load_state_dict(blk_weights, strict=False)
            print("[ASM] Loaded transformer blocks (strict=False).")
        if freeze_loaded:
            for p in actor.blocks.parameters():
                p.requires_grad = False
            print("[ASM] Froze transformer blocks.")
    else:
        print("[ASM] No weights matched transformer_blocks prefix; skipped.")

    print("[ASM] Warm-start complete. Slot embeddings & policy heads remain trainable.")


@dataclass
class Args:
    exp_name: str = "awac_offline"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "interact"
    wandb_entity: Optional[str] = None
    runs_dir: str = "runs"
    save_trained_model: bool = True
    model_save_dir: str = "trained_awac_models"

    domain_name: str = "cheetah"
    task_name: str = "run"
    dataset_level: str = "medium-expert"
    dataset_root_dir: str = 'datasets'

    # Shared nets
    batch_size: int = 256
    learning_rate: float = 3e-4
    lr_decay: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    bc_alpha: float = 0.0
    critic_hidden_dim: int = 256
    value_hidden_dim: int = 256

    # Actor selection
    actor_type: str = "saint"   # "mlp", "saint", "ar"

    # MLP
    actor_hidden_dim: int = 256
    actor_num_hidden_layers: int = 2
    actor_dropout_p: float = 0.1
    # AR
    actor_state_embed_dim: int = 128
    actor_action_embed_dim: int = 32
    actor_lstm_hidden_dim: int = 256
    actor_lstm_layers: int = 2
    # SAINT
    actor_d_model: int = 256
    actor_num_heads: int = 4
    actor_num_tf_blocks: int = 3
    actor_num_state_embeddings_M: int = 1
    actor_slot_emb_dropout_p: float = 0.0
    actor_transformer_dropout_p: float = 0.1
    actor_decision_head_hidden_size: int = 128
    # SPIN
    asm_model_path: Optional[str] = None
    freeze_asm_loaded_parts: bool = True

    # Training & Eval
    num_gradient_steps: int = int(1e6)
    eval_freq: int = 5000
    eval_episodes: int = 10

    # AWAC specifics
    awac_lambda: float = 0.3
    weight_clip: float = 100.0
    normalize_weights: bool = False

    state_dim: int = field(init=False)
    action_cardinalities: Tuple[int, ...] = field(default_factory=tuple)
    num_sub_actions_N: int = field(init=False)
    eval_max_episode_steps: int = field(init=False, default=1000)


def evaluate_policy(agent: Agent_AWAC, env_name: str, task_name: str, seed: int,
                    eval_eps: int, max_steps_per_episode: int):
    agent.actor.eval()
    eval_env = make_env(task_name=env_name, task=task_name, factorised=True)
    rewards = []
    for ep_idx in range(eval_eps):
        try:
            state_np, info = eval_env.reset(seed=seed + 100 + ep_idx)
        except TypeError:
            state_np = eval_env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state_np)
            state_np, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += reward
            steps += 1
        rewards.append(ep_ret)
    eval_env.close()
    agent.actor.train()
    avg = float(np.mean(rewards)) if rewards else 0.0
    std = float(np.std(rewards)) if rewards else 0.0
    print(f"Eval: {eval_eps} eps: Avg Reward {avg:.3f} +/- {std:.3f}")
    return avg, std


if __name__ == "__main__":
    args = tyro.cli(Args)
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

    print(f"Determining environment specs for {args.domain_name}-{args.task_name}...")
    tmp_env = make_env(task_name=args.domain_name, task=args.task_name, factorised=True)
    args.state_dim = tmp_env.observation_space.shape[0]
    if not hasattr(tmp_env.action_space, 'nvec'):
        raise ValueError("Env action space missing 'nvec' for factored actions.")
    args.action_cardinalities = tuple(tmp_env.action_space.nvec.tolist())
    args.num_sub_actions_N = len(args.action_cardinalities)
    args.eval_max_episode_steps = getattr(tmp_env, '_max_episode_steps', getattr(getattr(tmp_env, 'spec', None), 'max_episode_steps', 1000))
    tmp_env.close()
    print(f"State dim: {args.state_dim}, Action cards: {args.action_cardinalities}")

    buffer = OfflineReplayBuffer(device=device, env_name=args.domain_name, task_name=args.task_name,
                                 dataset_level=args.dataset_level, dataset_root_dir=args.dataset_root_dir)

    actor_type = args.actor_type.lower()

    actor_map = {
        "mlp": ActorMLP,
        "saint": ActorSAINT,
        "ar": ActorAR,
    }
    ActorCls = actor_map.get(actor_type)
    if ActorCls is None:
        raise ValueError("actor_type must be one of: mlp, saint, ar")

    if actor_type == "mlp":
        actor_kwargs = dict(
            hidden_dim=args.actor_hidden_dim,
            num_hidden_layers=args.actor_num_hidden_layers,
            dropout_p=args.actor_dropout_p,
            device=device,
        )
    elif actor_type == "saint":
        actor_kwargs = dict(
            actor_d_model=args.actor_d_model,
            actor_num_heads=args.actor_num_heads,
            actor_num_tf_blocks=args.actor_num_tf_blocks,
            actor_num_state_embeddings_M=args.actor_num_state_embeddings_M,
            actor_slot_emb_dropout_p=args.actor_slot_emb_dropout_p,
            actor_transformer_dropout_p=args.actor_transformer_dropout_p,
            actor_decision_head_hidden_size=args.actor_decision_head_hidden_size,
            actor_dropout_p=args.actor_dropout_p,
            device=device,
        )
    else:
        actor_kwargs = dict(
            state_embed_dim=args.actor_state_embed_dim,
            action_embed_dim=args.actor_action_embed_dim,
            lstm_hidden_dim=args.actor_lstm_hidden_dim,
            lstm_layers=args.actor_lstm_layers,
            device=device,
        )

    agent = Agent_AWAC(
        state_dim=args.state_dim,
        action_cardinalities=args.action_cardinalities,
        actor_ctor=ActorCls,
        actor_kwargs=actor_kwargs,
        critic_hidden_dim=args.critic_hidden_dim,
        value_hidden_dim=args.value_hidden_dim,
        bc_alpha=args.bc_alpha,
        awac_lambda=args.awac_lambda,
        weight_clip=args.weight_clip,
        normalize_weights=args.normalize_weights,
        lr=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        lr_decay=args.lr_decay,
        device=device,
    )

    if args.actor_type.lower() == "saint" and args.asm_model_path:
        if os.path.exists(args.asm_model_path):
            load_asm_weights_into_actor_saint(
                agent.actor, args.asm_model_path, device, args.freeze_asm_loaded_parts
            )
            run_name = f"AWAC-SPIN-{args.domain_name}__{args.seed}__{int(time.time())}"
        else:
            print(f"[ASM] Warning: asm_model_path does not exist: {args.asm_model_path}. Training from scratch.")
            run_name = f"AWAC-{args.actor_type}-{args.domain_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = f"AWAC-{args.actor_type}-{args.domain_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=asdict(args), name=run_name, save_code=True)
    writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k,v in asdict(args).items()]))

    print(f"Starting training for {args.num_gradient_steps} gradient steps...")
    start = time.time()
    for step in range(args.num_gradient_steps):
        s, a, r, s2, d = buffer.sample(args.batch_size)
        log = agent.train(s, a, r, s2, d)

        if step % 100 == 0:
            for k, v in log.items(): writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("misc/time_per_100_steps", (time.time() - start) / 100, step)
            start = time.time()

        if step % args.eval_freq == 0:
            print(f"\n--- Evaluating at step {step} ---")
            avg, std = evaluate_policy(agent, args.domain_name, args.task_name, args.seed, args.eval_episodes, args.eval_max_episode_steps)
            writer.add_scalar("eval/avg_reward", avg, step)
            writer.add_scalar("eval/std_reward", std, step)
            if args.save_trained_model:
                save_dir = os.path.join(args.model_save_dir, run_name)
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"model_step_{step}.pt")
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'value_state_dict': agent.value_net.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_opt.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_opt.state_dict(),
                    'value_optimizer_state_dict': agent.value_opt.state_dict(),
                    'args': asdict(args),
                    'step': step
                }, model_path)
                print(f"Saved checkpoint to {model_path}")

    if args.save_trained_model:
        save_dir = os.path.join(args.model_save_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"model_final.pt")
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'value_state_dict': agent.value_net.state_dict(),
            'actor_optimizer_state_dict': agent.actor_opt.state_dict(),
            'critic_optimizer_state_dict': agent.critic_opt.state_dict(),
            'value_optimizer_state_dict': agent.value_opt.state_dict(),
            'args': asdict(args),
            'step': args.num_gradient_steps
        }, model_path)
        print(f"Saved final model to {model_path}")
    writer.close()
    print("--- Training complete. ---")
