"""
SF1 factor return prediction with causal transformer.
Adapts the autoresearch GPT architecture for noisy numerical timeseries.
Predicts close-to-close returns (ret_cc) from 10 Sharadar SF1 factors.
Usage: uv run train_sf1.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

PARQUET_PATH = Path("~/claude_projects/sf1_models/sf1_shortlist_with_returns.parquet").expanduser()
FEATURE_COLS = [
    "bp", "ep", "fcf_yield", "gp_over_assets", "roic",
    "accruals", "log_marketcap", "asset_growth_yoy", "net_issuance",
    "days_since_datekey",
]
TARGET_COL = "ret_cc"
N_FEATURES = len(FEATURE_COLS)
VAL_CUTOFF = pd.Timestamp("2017-01-01")

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
SEQ_LEN = 64            # context window in trading days (~3 months)
N_LAYER = 2             # transformer layers
N_HEAD = 2              # attention heads (head_dim = N_EMBD // N_HEAD)
N_EMBD = 64             # model dimension
DROPOUT = 0.2           # dropout rate (regularization for noisy targets)

# Optimization
BATCH_SIZE = 256        # sequences per gradient step
LR = 3e-4               # peak learning rate
WEIGHT_DECAY = 0.01     # AdamW weight decay
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.05     # LR warmup fraction
WARMDOWN_RATIO = 0.3    # LR cosine decay fraction
FINAL_LR_FRAC = 0.0     # final LR as fraction of peak

# Loss
HUBER_DELTA = 0.02      # ~1 std of daily returns; clips outlier gradients

# Training
TIME_BUDGET = 300       # wall-clock training seconds

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    n_features: int = N_FEATURES
    sequence_len: int = SEQ_LEN
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    dropout: float = DROPOUT


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.resid_dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Continuous features → model dimension (replaces token embedding)
        self.input_proj = nn.Linear(config.n_features, config.n_embd, bias=False)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Scalar regression head (replaces vocab logits)
        self.output_head = nn.Linear(config.n_embd, 1, bias=True)
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rope(config.sequence_len * 10, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rope(self, seq_len, head_dim, base=10000, device=None):
        device = device or "cpu"
        freqs = torch.outer(
            torch.arange(seq_len, dtype=torch.float32, device=device),
            1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)),
        )
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        n = self.config.n_embd
        s = 3**0.5 * n**-0.5
        nn.init.uniform_(self.input_proj.weight, -s, s)
        nn.init.normal_(self.output_head.weight, std=0.001)
        nn.init.zeros_(self.output_head.bias)
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rope(
            self.config.sequence_len * 10, head_dim,
            device=self.input_proj.weight.device,
        )
        self.cos, self.sin = cos, sin

    def forward(self, features, targets=None):
        """
        features: (B, T, N_FEATURES) float
        targets:  (B, T) float returns, or None
        """
        B, T, _ = features.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.input_proj(features)
        x = self.input_dropout(x)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin)
        x = norm(x)

        preds = self.output_head(x).squeeze(-1)  # (B, T)

        if targets is not None:
            loss = F.huber_loss(preds, targets, delta=HUBER_DELTA)
            return loss, preds
        return preds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(seq_len, device):
    print(f"Loading {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    # Standardize all features using training-period statistics
    train_mask = df["date"] < VAL_CUTOFF
    feat_mean = df.loc[train_mask, FEATURE_COLS].mean().values.astype(np.float32)
    feat_std = np.clip(
        df.loc[train_mask, FEATURE_COLS].std().values.astype(np.float32), 1e-8, None
    )

    cutoff_np = np.datetime64(VAL_CUTOFF)
    ticker_info = []
    n_train_total = n_val_total = 0

    for _, group in df.groupby("ticker"):
        group = group.sort_values("date")
        n = len(group)
        n_windows = max(0, n - seq_len + 1)
        if n_windows == 0:
            continue
        dates = group["date"].values
        cutoff_idx = int(np.searchsorted(dates, cutoff_np))
        n_train = max(0, min(cutoff_idx - seq_len + 1, n_windows))
        n_val = n_windows - n_train
        feats = (group[FEATURE_COLS].values.astype(np.float32) - feat_mean) / feat_std
        tgts = group[TARGET_COL].values.astype(np.float32)
        ticker_info.append((feats, tgts, n_train, n_val))
        n_train_total += n_train
        n_val_total += n_val

    print(f"Building {n_train_total:,} train + {n_val_total:,} val windows (seq_len={seq_len})")

    train_features = np.empty((n_train_total, seq_len, N_FEATURES), dtype=np.float32)
    train_targets = np.empty((n_train_total, seq_len), dtype=np.float32)
    val_features = np.empty((n_val_total, seq_len, N_FEATURES), dtype=np.float32)
    val_targets = np.empty((n_val_total, seq_len), dtype=np.float32)

    ti = vi = 0
    win_idx = np.arange(seq_len)[None, :]
    for feats, tgts, n_train, n_val in ticker_info:
        n_windows = n_train + n_val
        idx = win_idx + np.arange(n_windows)[:, None]
        all_f, all_t = feats[idx], tgts[idx]
        if n_train > 0:
            train_features[ti:ti + n_train] = all_f[:n_train]
            train_targets[ti:ti + n_train] = all_t[:n_train]
            ti += n_train
        if n_val > 0:
            val_features[vi:vi + n_val] = all_f[n_train:]
            val_targets[vi:vi + n_val] = all_t[n_train:]
            vi += n_val

    train_f = torch.from_numpy(train_features).to(device)
    train_t = torch.from_numpy(train_targets).to(device)
    val_f = torch.from_numpy(val_features).to(device)
    val_t = torch.from_numpy(val_targets).to(device)
    print(f"Data on {device}: train {train_f.shape}, val {val_f.shape}")
    return train_f, train_t, val_f, val_t


def make_train_iter(features, targets, batch_size):
    """Infinite shuffled batch generator. Data already on device."""
    n = len(features)
    while True:
        perm = torch.randperm(n, device=features.device)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            yield features[idx], targets[idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_f, val_t, batch_size):
    model.eval()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    all_preds, all_tgts = [], []
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(val_f), batch_size):
        bf, bt = val_f[i:i + batch_size], val_t[i:i + batch_size]
        with autocast:
            loss, preds = model(bf, bt)
        total_loss += loss.item()
        n_batches += 1
        # Last position: most context, no duplication across overlapping windows
        all_preds.append(preds[:, -1].float())
        all_tgts.append(bt[:, -1].float())

    p = torch.cat(all_preds)
    t = torch.cat(all_tgts)
    mse = F.mse_loss(p, t).item()

    # Pearson correlation
    pc = p - p.mean()
    tc = t - t.mean()
    pearson = (pc * tc).sum() / (pc.norm() * tc.norm() + 1e-8)

    # Spearman rank correlation (IC)
    pr = p.argsort().argsort().float()
    tr = t.argsort().argsort().float()
    prc, trc = pr - pr.mean(), tr - tr.mean()
    spearman = (prc * trc).sum() / (prc.norm() * trc.norm() + 1e-8)

    model.train()
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "pearson": pearson.item(),
        "spearman_ic": spearman.item(),
        "huber_loss": total_loss / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

train_f, train_t, val_f, val_t = load_data(SEQ_LEN, device)

config = ModelConfig()
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = TimeSeriesTransformer(config)
model.to_empty(device=device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    [
        {"params": model.input_proj.parameters(), "lr": LR},
        {"params": model.blocks.parameters(), "lr": LR},
        {"params": model.output_head.parameters(), "lr": LR * 0.1},
        {"params": [model.resid_lambdas], "lr": LR * 0.01},
        {"params": [model.x0_lambdas], "lr": LR * 0.1},
    ],
    weight_decay=WEIGHT_DECAY,
    betas=ADAM_BETAS,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

model = torch.compile(model, dynamic=False)

train_iter = make_train_iter(train_f, train_t, BATCH_SIZE)
x, y = next(train_iter)

print(f"Time budget: {TIME_BUDGET}s | Batch size: {BATCH_SIZE}")


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_train_start = time.time()
smooth_loss = 0.0
total_training_time = 0.0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    with autocast_ctx:
        loss, _ = model(x, y)
    loss.backward()

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    x, y = next(train_iter)
    loss_f = loss.item()

    if math.isnan(loss_f) or loss_f > 100:
        print("\nFAIL: loss exploded")
        exit(1)

    torch.cuda.synchronize()
    dt = time.time() - t0

    if step > 5:
        total_training_time += dt

    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_f
    debiased = smooth_loss / (1 - ema ** (step + 1))
    remaining = max(0, TIME_BUDGET - total_training_time)

    if step % 100 == 0:
        print(
            f"\rstep {step:05d} ({100 * progress:.1f}%) | loss: {debiased:.6f} "
            f"| lrm: {lrm:.4f} | dt: {dt * 1000:.0f}ms | remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
metrics = evaluate(model, val_f, val_t, BATCH_SIZE)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_mse:          {metrics['mse']:.8f}")
print(f"val_rmse:         {metrics['rmse']:.6f}")
print(f"val_pearson:      {metrics['pearson']:.6f}")
print(f"val_spearman_ic:  {metrics['spearman_ic']:.6f}")
print(f"val_huber_loss:   {metrics['huber_loss']:.8f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
print(f"n_layer:          {N_LAYER}")
print(f"n_embd:           {N_EMBD}")
print(f"seq_len:          {SEQ_LEN}")
