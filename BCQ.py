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


class ImitationMLP(nn.Module):
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
    def sample_actions(self, states: torch.Tensor) -> torch.Tensor:
        h = self.shared(states)
        outs = []
        for head in self.heads:
            logits = head(h)
            outs.append(torch.distributions.Categorical(logits=logits).sample())
        return torch.stack(outs, dim=1)


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
        B, Nq, _ = Q.shape
        _, Nk, _ = K.shape
        Qn = self.ln_q(Q)
        Kn = self.ln_kv(K)
        Vn = self.ln_kv(K if V_in is None else V_in)
        q = self.fc_q(Qn).view(B, Nq, self.num_heads, -1).permute(0,2,1,3)
        k = self.fc_k(Kn).view(B, Nk, self.num_heads, -1).permute(0,2,3,1)
        v = self.fc_v(Vn).view(B, Nk, self.num_heads, -1).permute(0,2,1,3)
        attn = torch.matmul(q, k) / (q.size(-1)**0.5)
        if attention_mask is not None: attn = attn.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        p = torch.softmax(attn, dim=-1)
        p = self.attn_dropout(p)
        out = torch.matmul(p, v).permute(0,2,1,3).contiguous().view(B, Nq, -1)
        x = Q + self.out_dropout(self.out_proj(out))
        x = x + self.ffn(self.ffn_ln(x))
        return x, p


class SABMod(nn.Module):
    def __init__(self, dim_in_out, num_heads, ffn_hidden_mult=2, dropout_p=0.1):
        super().__init__()
        self.mab = MABMod(dim_in_out, dim_in_out, dim_in_out, num_heads, ffn_hidden_mult, dropout_p)

    def forward(self, X, attention_mask=None):
        return self.mab(X, X, V_in=X, attention_mask=attention_mask)


class ImitationSAINT(nn.Module):
    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...],
                 d_model: int = 256, num_heads: int = 4, num_tf_blocks: int = 3,
                 num_state_embeddings_M: int = 1, slot_emb_dropout_p: float = 0.0,
                 transformer_dropout_p: float = 0.1, decision_head_hidden_size: int = 128,
                 head_dropout_p: float = 0.1, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.N = len(action_cardinalities)
        self.cards = action_cardinalities
        self.D = d_model
        self.M = num_state_embeddings_M
        self.device = device

        self.state_encoder = None
        if self.M > 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, self.D * 2), nn.GELU(),
                nn.Linear(self.D * 2, self.M * self.D)
            ).to(device)

        self.slot_embeddings = nn.Parameter(torch.randn(self.N, self.D))
        self.slot_dropout = nn.Dropout(slot_emb_dropout_p)
        self.blocks = nn.ModuleList([SABMod(self.D, num_heads, 2, transformer_dropout_p) for _ in range(num_tf_blocks)]).to(device)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.D, decision_head_hidden_size), nn.GELU(),
                          nn.Dropout(head_dropout_p), nn.Linear(decision_head_hidden_size, c)).to(device)
            for c in self.cards
        ])

    def _prep(self, s: torch.Tensor):
        B = s.size(0)
        slots = self.slot_dropout(self.slot_embeddings.unsqueeze(0).repeat(B,1,1))
        if self.M > 0 and self.state_encoder is not None:
            st = self.state_encoder(s).view(B, self.M, self.D)
            return torch.cat([st, slots], dim=1)
        return slots

    def _pass(self, s: torch.Tensor):
        x = self._prep(s)
        for blk in self.blocks: x, _ = blk(x)
        return x[:, self.M:, :] if self.M > 0 else x

    def forward(self, states: torch.Tensor) -> List[torch.Tensor]:
        ctx = self._pass(states)
        out = []
        for i, head in enumerate(self.heads):
            logits = head(ctx[:, i, :])
            out.append(F.log_softmax(logits, dim=-1))
        return out

    @torch.no_grad()
    def sample_actions(self, states: torch.Tensor) -> torch.Tensor:
        ctx = self._pass(states)
        outs = []
        for i, head in enumerate(self.heads):
            logits = head(ctx[:, i, :])
            outs.append(torch.distributions.Categorical(logits=logits).sample())
        return torch.stack(outs, dim=1)


def load_asm_weights_into_saint(
    model: ImitationSAINT,
    asm_ckpt_path: str,
    device: torch.device,
    freeze_loaded: bool,
    *,
    state_encoder_prefix: str = "core_transformer.state_encoder.",
    tf_blocks_prefix: str = "core_transformer.transformer_blocks."
):
    print(f"[ASM] Loading ASM checkpoint: {asm_ckpt_path}")
    if not os.path.exists(asm_ckpt_path): raise FileNotFoundError(asm_ckpt_path)
    ckpt = torch.load(asm_ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        print("[ASM] No model_state_dict; skip.")
        return
    src = ckpt["model_state_dict"]

    # state encoder
    if model.state_encoder is not None:
        se = {k.replace(state_encoder_prefix, ""): v for k, v in src.items() if k.startswith(state_encoder_prefix)}
        if se:
            try:
                model.state_encoder.load_state_dict(se, strict=True)
                print("[ASM] Loaded state_encoder strict.")
            except RuntimeError as e:
                print(f"[ASM] strict failed: {e} → loose")
                model.state_encoder.load_state_dict(se, strict=False)
            if freeze_loaded:
                for p in model.state_encoder.parameters(): p.requires_grad = False
                print("[ASM] Froze state_encoder.")
    # blocks
    blk = {k.replace(tf_blocks_prefix, ""): v for k, v in src.items() if k.startswith(tf_blocks_prefix)}
    if blk:
        try:
            model.blocks.load_state_dict(blk, strict=True)
            print("[ASM] Loaded blocks strict.")
        except RuntimeError as e:
            print(f"[ASM] strict failed: {e} → loose")
            model.blocks.load_state_dict(blk, strict=False)
        if freeze_loaded:
            for p in model.blocks.parameters(): p.requires_grad = False
            print("[ASM] Froze blocks.")
    print("[ASM] Warm-start complete. Heads/slots remain trainable.")


class ImitationAR(nn.Module):
    autoregressive = True

    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...],
                 state_embed_dim: int = 128, action_embed_dim: int = 32,
                 lstm_hidden_dim: int = 256, lstm_layers: int = 2,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.N = len(action_cardinalities)
        self.cards = action_cardinalities
        self.device = device
        self.state_embed = nn.Sequential(nn.Linear(state_dim, state_embed_dim), nn.ReLU()).to(device)

        self.max_card = max(self.cards)
        self.start_token_idx = self.max_card
        self.action_embed = nn.Embedding(self.max_card + 1, action_embed_dim).to(device)

        self.lstm = nn.LSTM(state_embed_dim + action_embed_dim, lstm_hidden_dim, lstm_layers, batch_first=True).to(device)
        self.heads = nn.ModuleList([nn.Linear(lstm_hidden_dim, c).to(device) for c in self.cards])
        self.lstm_layers = lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim

    def _init_hidden(self, B):
        h0 = torch.zeros(self.lstm_layers, B, self.lstm_hidden_dim, device=self.device)
        return (h0, torch.zeros_like(h0))

    def teacher_forcing_logps(self, s: torch.Tensor, a: torch.Tensor) -> List[torch.Tensor]:
        B = s.size(0)
        se = self.state_embed(s)
        h = self._init_hidden(B)
        prev = self.action_embed(torch.full((B,), self.start_token_idx, dtype=torch.long, device=self.device))
        outs = []
        for i in range(self.N):
            x = torch.cat([se, prev], dim=1).unsqueeze(1)
            y, h = self.lstm(x, h)
            logits = self.heads[i](y.squeeze(1))
            outs.append(F.log_softmax(logits, dim=-1))
            prev = self.action_embed(a[:, i])
        return outs

    @torch.no_grad()
    def sample_actions(self, s: torch.Tensor) -> torch.Tensor:
        B = s.size(0)
        se = self.state_embed(s)
        h = self._init_hidden(B)
        prev = self.action_embed(torch.full((B,), self.start_token_idx, dtype=torch.long, device=self.device))
        samples = []

        for i in range(self.N):
            x = torch.cat([se, prev], dim=1).unsqueeze(1)
            y, h = self.lstm(x, h)
            logits = self.heads[i](y.squeeze(1))
            ai = torch.distributions.Categorical(logits=logits).sample()
            samples.append(ai)
            prev = self.action_embed(ai)
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def sample_actions_K(self, s: torch.Tensor, K: int) -> torch.Tensor:
        B = s.size(0)
        se = self.state_embed(s)
        allK = []
        for _ in range(K):
            h = self._init_hidden(B)
            prev = self.action_embed(torch.full((B,), self.start_token_idx, dtype=torch.long, device=self.device))
            samples = []
            for i in range(self.N):
                x = torch.cat([se, prev], dim=1).unsqueeze(1)
                y, h = self.lstm(x, h)
                logits = self.heads[i](y.squeeze(1))
                ai = torch.distributions.Categorical(logits=logits).sample()
                samples.append(ai)
                prev = self.action_embed(ai)
            allK.append(torch.stack(samples, dim=1))
        return torch.stack(allK, dim=1)


try:
    from dmc_datasets.environment_utils import make_env
except ImportError:
    print("Fatal: dmc_datasets.environment_utils.make_env not found.")
    raise


class OfflineReplayBuffer:
    def __init__(self, device: torch.device, env_name: str, task_name: str,
                 dataset_level: str, dataset_root_dir: str, bin_size: int):
        self.device = device
        env = make_env(task_name=env_name, task=task_name, factorised=True, bin_size=bin_size)
        raw = env.load_dataset(level=dataset_level, data_dir=dataset_root_dir)
        env.close()

        if not raw:
            raise ValueError("Loaded dataset is empty.")
        obs  = np.array([t[0] for t in raw], dtype=np.float32)
        acts = np.array([t[1] for t in raw], dtype=np.int64)
        rews = np.array([t[2] for t in raw], dtype=np.float32).reshape(-1,1)
        nxt  = np.array([t[3] for t in raw], dtype=np.float32)
        dones= np.array([t[4] for t in raw], dtype=np.float32).reshape(-1,1)
        self.states = torch.from_numpy(obs).to(device)
        self.actions = torch.from_numpy(acts).to(device)
        self.rewards = torch.from_numpy(rews).to(device)
        self.next_states = torch.from_numpy(nxt).to(device)
        self.dones = torch.from_numpy(dones).to(device)
        self.size = self.states.shape[0]
    def sample(self, batch):
        idx = torch.randint(0, self.size, (batch,), device=self.states.device)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]


# --- Critics ---
class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for e in range(self.weight.size(0)):
            nn.init.kaiming_uniform_(self.weight[e], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1/math.sqrt(fan_in) if fan_in>0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight + self.bias


class VectorizedDQN(nn.Module):
    def __init__(self, state_dim, action_sum: int, num_critics=2, hidden=256):
        super().__init__()
        self.E = num_critics
        self.l1 = VectorizedLinear(state_dim, hidden, self.E)
        self.l2 = VectorizedLinear(hidden, hidden, self.E)
        self.qs = VectorizedLinear(hidden, action_sum, self.E)

    def forward(self, s, cards: Tuple[int, ...]):
        se = s.unsqueeze(0).repeat_interleave(self.E, dim=0)
        qf = F.relu(self.l1(se))
        qf = F.relu(self.l2(qf))
        qf = self.qs(qf)
        outs, idx = [], 0
        for c in cards:
            outs.append(qf[:, :, idx:idx+c])
            idx += c
        return outs


# --- Agent ---
class Agent_BCQ:
    def __init__(self, state_dim: int, action_cardinalities: Tuple[int, ...],
                 imitation_module: nn.Module,
                 num_critics: int = 2, critic_hidden_dim: int = 256,
                 bcq_threshold: float = 0.3, ar_candidates_K: int = 10,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, lr_decay: float = 1.0,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.cards = action_cardinalities
        self.N = len(self.cards)
        self.gamma, self.tau = gamma, tau
        self.E = num_critics
        self.imitation = imitation_module.to(device)
        self.bcq_log_tau = torch.log(torch.tensor([bcq_threshold], device=device))
        self.K = ar_candidates_K

        qdim = sum(self.cards)
        self.critic = VectorizedDQN(state_dim, qdim, num_critics, critic_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.im_opt = torch.optim.Adam(self.imitation.parameters(), lr=lr)
        self.cr_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.im_sched = ExponentialLR(self.im_opt, gamma=lr_decay)
        self.cr_sched = ExponentialLR(self.cr_opt, gamma=lr_decay)

    @torch.no_grad()
    def _bcq_mask_select(self, s: torch.Tensor) -> torch.Tensor:
        q_list = self.critic(s, self.cards)
        logps = self.imitation(s)
        out = []
        for i in range(self.N):
            qi = q_list[i].min(0).values
            lpi = logps[i]
            max_lp = lpi.max(dim=1, keepdim=True).values
            mask = (lpi >= self.bcq_log_tau + max_lp)
            qi = qi.clone()
            qi[~mask] = -1e8
            out.append(qi.argmax(dim=1))
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def _bcq_candidates_select(self, s: torch.Tensor) -> torch.Tensor:
        aK = self.imitation.sample_actions_K(s, self.K)
        B = s.size(0)
        Srep = s.unsqueeze(1).repeat(1, self.K, 1).view(B*self.K, -1)
        Arep = aK.view(B*self.K, self.N)
        q1, q2 = [], []
        q_list = self.critic(Srep, self.cards)

        q_sel = []
        for i in range(self.N):
            idx = Arep[:, i].unsqueeze(0).unsqueeze(-1).expand(self.E, -1, 1)
            qi = torch.gather(q_list[i], 2, idx).squeeze(-1)
            q_sel.append(qi)
        q_sum = torch.stack(q_sel, dim=0).sum(dim=0)
        q_min = q_sum.min(0).values
        q_min = q_min.view(B, self.K)
        best_idx = q_min.argmax(dim=1)
        return aK[torch.arange(B, device=s.device), best_idx, :]

    @torch.no_grad()
    def select_action(self, state_np: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if getattr(self.imitation, "autoregressive", False):
            a = self._bcq_candidates_select(s)
        else:
            a = self._bcq_mask_select(s)
        return a.squeeze(0).cpu().numpy()

    def train(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, s2: torch.Tensor, d: torch.Tensor):
        log = {}
        # ----- Critic update -----
        with torch.no_grad():
            if getattr(self.imitation, "autoregressive", False):
                aK = self.imitation.sample_actions_K(s2, self.K)
                B = s2.size(0)
                Srep = s2.unsqueeze(1).repeat(1, self.K, 1).view(B*self.K, -1)
                Arep = aK.view(B*self.K, self.N)
                q_list_t = self.critic_target(Srep, self.cards)
                q_sel = []
                for i in range(self.N):
                    idx = Arep[:, i].unsqueeze(0).unsqueeze(-1).expand(self.E, -1, 1)
                    qi = torch.gather(q_list_t[i], 2, idx).squeeze(-1)
                    q_sel.append(qi)
                q_sum = torch.stack(q_sel, dim=0).sum(dim=0)
                q_min = q_sum.min(0).values.view(B, self.K)
                next_q = q_min.max(dim=1).values
            else:
                logps_next = self.imitation(s2)
                q_list_t = self.critic_target(s2, self.cards)
                q_sum_t = torch.zeros(self.E, s2.size(0), device=self.device)
                for i in range(self.N):
                    lpi = logps_next[i]
                    max_lp = lpi.max(dim=1, keepdim=True).values
                    mask = (lpi >= self.bcq_log_tau + max_lp)
                    maskE = mask.unsqueeze(0).expand(self.E, -1, -1)
                    qi = q_list_t[i].clone()
                    qi[~maskE] = -1e8
                    max_q_i, _ = qi.max(dim=2)
                    q_sum_t += max_q_i
                next_q = q_sum_t.min(0).values

            y = r.squeeze(-1) + (1.0 - d.squeeze(-1)) * self.gamma * next_q
            y = y.unsqueeze(0).expand(self.E, -1)

        q_list = self.critic(s, self.cards)
        q_sel = []
        for i in range(self.N):
            idx = a[:, i].unsqueeze(0).unsqueeze(-1).expand(self.E, -1, 1)
            qi = torch.gather(q_list[i], 2, idx).squeeze(-1)
            q_sel.append(qi)
        q_cur = torch.stack(q_sel, dim=0).sum(dim=0)
        critic_loss = F.mse_loss(q_cur, y)
        self.cr_opt.zero_grad()
        critic_loss.backward()
        self.cr_opt.step()
        self.cr_sched.step()
        log['critic_loss'] = critic_loss.item()

        if getattr(self.imitation, "autoregressive", False):
            logps = self.imitation.teacher_forcing_logps(s, a)
        else:
            logps = self.imitation(s)
        sum_logp = 0.0
        for i in range(self.N):
            sum_logp = sum_logp + torch.gather(logps[i], 1, a[:, i].unsqueeze(-1)).squeeze(-1)
        im_loss = -(sum_logp).mean()
        self.im_opt.zero_grad()
        im_loss.backward()
        self.im_opt.step()
        self.im_sched.step()
        log['imitation_loss'] = im_loss.item()

        # ----- Polyak -----
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return log


@dataclass
class ArgsBCQ:
    exp_name: str = "bcq_offline"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "interact"
    wandb_entity: Optional[str] = None
    runs_dir: Optional[str] = None
    save_trained_model: bool = True
    model_save_dir: str = "trained_BCQ_models"

    domain_name: str = "cheetah"
    task_name: str = "run"
    dataset_level: str = "medium-expert"
    dataset_root_dir: str = 'datasets'
    bin_size: int = 3

    # imitation selection
    imit_type: str = "mlp"  # mlp | saint | ar

    # MLP
    imitation_hidden_dim: int = 256
    imitation_num_hidden_layers: int = 2
    imitation_dropout_p: float = 0.0

    # SAINT
    saint_d_model: int = 256
    saint_num_heads: int = 4
    saint_num_tf_blocks: int = 3
    saint_num_state_embeddings_M: int = 1
    saint_slot_emb_dropout_p: float = 0.0
    saint_transformer_dropout_p: float = 0.1
    saint_decision_head_hidden_size: int = 128
    saint_head_dropout_p: float = 0.1
    asm_model_path: Optional[str] = None
    freeze_asm_loaded_parts: bool = True

    # AR
    ar_state_embed_dim: int = 128
    ar_action_embed_dim: int = 32
    ar_lstm_hidden_dim: int = 256
    ar_lstm_layers: int = 2
    ar_candidates_K: int = 10

    # BCQ / opt
    batch_size: int = 256
    learning_rate: float = 3e-4
    lr_decay: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    bcq_threshold: float = 0.3
    num_critics: int = 2
    critic_hidden_dim: int = 256

    # train & eval
    num_gradient_steps: int = int(1e6)
    eval_freq: int = 5000
    eval_episodes: int = 10
    eval_max_episode_steps: int = field(init=False, default=1000)

    # filled later
    state_dim: int = field(init=False)
    action_cardinalities: Tuple[int, ...] = field(default_factory=tuple)
    num_sub_actions_N: int = field(init=False)

    def __post_init__(self):
        if not self.runs_dir:
            if self.bin_size == 3:
                self.runs_dir = os.path.join(os.getcwd(), "runs", self.dataset_level, self.domain_name)
            else:
                self.runs_dir = os.path.join(os.getcwd(), "runs", self.dataset_level, self.domain_name, str(self.bin_size))

def evaluate_policy(agent: Agent_BCQ, env_name: str, task_name: str, seed: int,
                    eval_eps: int, max_steps_per_episode: int, bin_size: int):
    agent.imitation.eval()
    eval_env = make_env(task_name=env_name, task=task_name, factorised=True, bin_size=bin_size)
    rewards = []
    for ep in range(eval_eps):
        try: s_np, _ = eval_env.reset(seed=seed + 100 + ep)
        except TypeError: s_np = eval_env.reset()
        done = False
        ret = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            a = agent.select_action(s_np)
            s_np, r, term, trunc, _ = eval_env.step(a)
            done = term or trunc
            ret += r
            steps += 1
        rewards.append(ret)
    eval_env.close()
    agent.imitation.train()
    avg = float(np.mean(rewards)) if rewards else 0.0
    std = float(np.std(rewards)) if rewards else 0.0
    print(f"Eval: {eval_eps} eps: Avg Reward {avg:.3f} +/- {std:.3f}")
    return avg, std


if __name__ == "__main__":
    args = tyro.cli(ArgsBCQ)
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

    tmp = make_env(task_name=args.domain_name, task=args.task_name, factorised=True, bin_size=args.bin_size)
    args.state_dim = tmp.observation_space.shape[0]
    if not hasattr(tmp.action_space, 'nvec'): raise ValueError("Env action space missing 'nvec'.")
    args.action_cardinalities = tuple(tmp.action_space.nvec.tolist())
    args.num_sub_actions_N = len(args.action_cardinalities)
    args.eval_max_episode_steps = getattr(tmp, '_max_episode_steps', getattr(getattr(tmp, 'spec', None), 'max_episode_steps', 1000))
    tmp.close()
    print(f"State dim: {args.state_dim}, Action cards: {args.action_cardinalities}")

    buf = OfflineReplayBuffer(device=device, env_name=args.domain_name, task_name=args.task_name,
                              dataset_level=args.dataset_level, dataset_root_dir=args.dataset_root_dir,
                              bin_size=args.bin_size)

    imit_type = args.imit_type.lower()
    if imit_type == "mlp":
        ImitCls = ImitationMLP
        imit = ImitCls(args.state_dim, args.action_cardinalities,
                       hidden_dim=args.imitation_hidden_dim,
                       num_hidden_layers=args.imitation_num_hidden_layers,
                       dropout_p=args.imitation_dropout_p, device=device)
    elif imit_type == "saint":
        ImitCls = ImitationSAINT
        imit = ImitCls(args.state_dim, args.action_cardinalities,
                       d_model=args.saint_d_model, num_heads=args.saint_num_heads, num_tf_blocks=args.saint_num_tf_blocks,
                       num_state_embeddings_M=args.saint_num_state_embeddings_M, slot_emb_dropout_p=args.saint_slot_emb_dropout_p,
                       transformer_dropout_p=args.saint_transformer_dropout_p, decision_head_hidden_size=args.saint_decision_head_hidden_size,
                       head_dropout_p=args.saint_head_dropout_p, device=device)
        if args.asm_model_path:
            if os.path.exists(args.asm_model_path):
                load_asm_weights_into_saint(imit, args.asm_model_path, device, args.freeze_asm_loaded_parts)
                print("[ASM] SAINT imitation warm-started.")
            else:
                print(f"[ASM] Warning: asm_model_path not found: {args.asm_model_path}.")
    elif imit_type == "ar":
        ImitCls = ImitationAR
        imit = ImitCls(args.state_dim, args.action_cardinalities,
                       state_embed_dim=args.ar_state_embed_dim, action_embed_dim=args.ar_action_embed_dim,
                       lstm_hidden_dim=args.ar_lstm_hidden_dim, lstm_layers=args.ar_lstm_layers, device=device)
    else:
        raise ValueError("imit_type must be one of: mlp, saint, ar")

    agent = Agent_BCQ(
        state_dim=args.state_dim, action_cardinalities=args.action_cardinalities,
        imitation_module=imit, num_critics=args.num_critics, critic_hidden_dim=args.critic_hidden_dim,
        bcq_threshold=args.bcq_threshold, ar_candidates_K=args.ar_candidates_K,
        lr=args.learning_rate, gamma=args.gamma, tau=args.tau, lr_decay=args.lr_decay, device=device
    )

    run_name = ("BCQ-SPIN-" if (imit_type=="saint" and args.asm_model_path and os.path.exists(args.asm_model_path)) else f"BCQ-{imit_type}-") \
               + f"{args.domain_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k,v in asdict(args).items()]))
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=asdict(args), name=run_name, save_code=True)

    print(f"Starting training for {args.num_gradient_steps} gradient steps...")
    t0 = time.time()
    for step in range(args.num_gradient_steps):
        s, a, r, s2, d = buf.sample(args.batch_size)
        log = agent.train(s, a, r, s2, d)

        if step % 100 == 0:
            for k, v in log.items(): writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("misc/time_per_100_steps", (time.time() - t0) / 100, step)
            t0 = time.time()

        if step % args.eval_freq == 0:
            print(f"\n--- Evaluating at step {step} ---")
            avg, std = evaluate_policy(agent, args.domain_name, args.task_name, args.seed,
                                       args.eval_episodes, args.eval_max_episode_steps, args.bin_size)
            writer.add_scalar("eval/avg_reward", avg, step)
            writer.add_scalar("eval/std_reward", std, step)
            if args.save_trained_model:
                save_dir = os.path.join(args.model_save_dir, run_name)
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"model_step_{step}.pt")
                torch.save({
                    'imitation_state_dict': agent.imitation.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'im_opt_state_dict': agent.im_opt.state_dict(),
                    'cr_opt_state_dict': agent.cr_opt.state_dict(),
                    'args': asdict(args), 'step': step
                }, path)
                print(f"Saved checkpoint to {path}")

    if args.save_trained_model:
        save_dir = os.path.join(args.model_save_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"model_final.pt")
        torch.save({
            'imitation_state_dict': agent.imitation.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'im_opt_state_dict': agent.im_opt.state_dict(),
            'cr_opt_state_dict': agent.cr_opt.state_dict(),
            'args': asdict(args), 'step': args.num_gradient_steps
        }, path)
        print(f"Saved final model to {path}")
    writer.close()
    if args.track and 'wandb' in locals() and wandb.run is not None: wandb.finish()
    print("--- Training complete. ---")
