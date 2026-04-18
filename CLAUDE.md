# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Autonomous pretraining research framework (by Karpathy). An AI agent iterates on a small GPT training setup: modify code, train for 5 minutes, check if val_bpb improved, keep or discard, repeat. The human programs `program.md` to guide research direction; the agent only edits `train.py`.

## Commands

```bash
# Setup (one-time)
uv sync                    # install dependencies
uv run prepare.py          # download data + train tokenizer (~2 min)

# Run a training experiment (always 5 min wall-clock)
uv run train.py > run.log 2>&1

# Extract results
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

No test suite or linter exists. The "test" is whether `train.py` runs without crashing and produces a lower `val_bpb`.

## Three Files That Matter

- **`prepare.py`** — Read-only. Data prep, tokenizer, dataloader, `evaluate_bpb()`. Fixed constants: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS=40*524288`, `VOCAB_SIZE=8192`.
- **`train.py`** — The only file the agent edits. Full GPT model, MuonAdamW optimizer, training loop. Hyperparameters are at the top (lines ~428-451), edited directly with no CLI flags.
- **`program.md`** — Agent instructions and experiment protocol. Human edits this to steer research.

## Architecture Overview

**Model**: Transformer GPT with RoPE, grouped query attention (2:1 KV grouping), value embeddings on alternating layers (ResFormer), sliding window attention (`WINDOW_PATTERN`), ReLU² activation, RMSNorm, learnable residual scaling.

**Optimizer (MuonAdamW)**: Hybrid — Muon (with Nesterov momentum + polar orthogonalization) for 2D matrix params, AdamW for embeddings/scalars. Four separate LR groups: `EMBEDDING_LR`, `UNEMBEDDING_LR`, `MATRIX_LR`, `SCALAR_LR`. All scaled by `(model_dim/768)^-0.5`.

**Model sizing**: `model_dim = DEPTH * ASPECT_RATIO`, rounded up to nearest `HEAD_DIM`. `DEPTH` is the primary complexity knob.

**Training loop**: Gradient accumulation over micro-steps, bfloat16 autocast, warmup/warmdown LR schedule, Muon momentum annealing (0.85→0.95), weight decay annealing to 0. Exits after `TIME_BUDGET` seconds of training.

**Metric**: `val_bpb` (bits per byte) — lower is better. Vocab-size-independent, computed on a fixed validation shard. Special tokens excluded.

## Experiment Protocol

See `program.md` for the full protocol. Key points:

- Branch naming: `autoresearch/<tag>`
- First run is always baseline (unmodified `train.py`)
- Results logged to `results.tsv` (tab-separated, 5 columns: commit/val_bpb/memory_gb/status/description)
- `results.tsv` stays untracked (not committed)
- If improved: keep the commit. If not: `git reset` to discard.
- Commit messages use `exp:` prefix (e.g., `exp: GQA n_kv_head=4->2`)
- **NEVER STOP**: once the loop starts, run indefinitely without asking

## Constraints

- Cannot modify `prepare.py` or install new packages
- Cannot change evaluation, tokenizer, data loading, or time budget
- Only `train.py` is edited
- VRAM is a soft constraint (some increase OK for meaningful gains)
- Simplicity matters: small improvement + ugly complexity = not worth it
