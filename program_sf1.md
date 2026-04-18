# autoresearch — SF1 factor return prediction

This is an experiment to have the LLM optimize a transformer for predicting stock returns from fundamental factors.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `sf1-apr17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train_sf1.py` — the file you modify. Model architecture, optimizer, training loop, data loading, evaluation.
4. **Verify data exists**: Check that `~/claude_projects/sf1_models/sf1_shortlist_with_returns.parquet` exists.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task

Predict daily close-to-close stock returns (`ret_cc`) from 10 Sharadar SF1 fundamental factors: `bp`, `ep`, `fcf_yield`, `gp_over_assets`, `roic`, `accruals`, `log_marketcap`, `asset_growth_yoy`, `net_issuance`, `days_since_datekey`.

The data is a panel of ~505 US stocks from 2010–2018 (~1M rows). Train on pre-2017 data, validate on 2017+. The primary metric is **val_spearman_ic** (Spearman rank correlation between predicted and actual returns) — higher is better. Secondary metrics: val_mse, val_pearson.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train_sf1.py`.

**What you CAN do:**
- Modify `train_sf1.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, data preprocessing, feature engineering, loss function, batch size, model size, sequence length, etc.

**What you CANNOT do:**
- Modify any other files in the repo.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the parquet data source or the train/val split date (2017-01-01).

**The goal is simple: get the highest val_spearman_ic.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the data pipeline. The only constraint is that the code runs without crashing and finishes within the time budget.

**Key considerations for this domain:**
- Financial returns are extremely noisy (signal-to-noise ratio ~0.02). Overfitting is the primary risk, not underfitting.
- Regularization matters more than model capacity: dropout, weight decay, small models, early stopping.
- Cross-sectional signal (how stocks rank relative to each other on a given day) matters more than absolute prediction accuracy.
- The features are mostly rank-normalized to [-1, 1] except `days_since_datekey` (0–180) and `log_marketcap`. All features are standardized during loading.
- Missing values (~1–2% per feature) are filled with 0 before standardization.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_mse:          0.00031234
val_rmse:         0.017674
val_pearson:      0.012345
val_spearman_ic:  0.011234
val_huber_loss:   0.00456789
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     1234.5
num_steps:        95000
num_params:       787,849
n_layer:          4
n_embd:           128
seq_len:          64
```

You can extract the key metric from the log file:

```
grep "^val_spearman_ic:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_spearman_ic	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_spearman_ic achieved (e.g. 0.012345) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 1.2 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_spearman_ic	memory_gb	status	description
a1b2c3d	0.011234	1.2	keep	baseline
b2c3d4e	0.013200	1.3	keep	increase dropout to 0.2
c3d4e5f	0.010000	1.2	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/sf1-apr17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_sf1.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train_sf1.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_spearman_ic:\|^val_pearson:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_spearman_ic improved (higher), you "advance" the branch, keeping the git commit
9. If val_spearman_ic is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, revisit hyperparameters. The loop runs until the human interrupts you, period.

## Ideas to explore

These are suggestions, not prescriptions. Use your judgment.

**Architecture:**
- Sequence length: try shorter (32) or longer (128, 256)
- Model size: more/fewer layers, wider/narrower embeddings
- Head count and head dimension
- Remove or modify RoPE
- Try bidirectional attention (non-causal) — future-peeking is OK if the model is predicting from today's features, not future features
- Different activations (GELU, SiLU instead of ReLU²)
- Add LayerNorm/BatchNorm at different points

**Regularization (likely most impactful for this domain):**
- Dropout rate (try 0.05–0.3)
- Weight decay
- Gradient clipping
- Feature noise injection during training
- Prediction clipping/capping

**Loss function:**
- MSE instead of Huber
- Huber delta tuning
- Asymmetric loss (weight positive/negative returns differently)
- Rank-based loss (directly optimize IC)

**Data pipeline:**
- Feature engineering: interactions, rolling statistics, cross-sectional ranks
- Different NaN handling (per-ticker forward-fill, learned imputation)
- Target normalization (standardize returns per date)
- Different train/val split date
- Non-overlapping windows (stride > 1) for less correlated training samples

**Optimizer:**
- Learning rate and schedule
- Different optimizer (SGD+momentum, AdaFactor, the MuonAdamW from train.py)
- Per-group learning rates
