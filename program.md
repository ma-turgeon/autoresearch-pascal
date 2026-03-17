# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Platform constraints (GTX 1080 Ti / Pascal / PyTorch 2.0.1)

**IMPORTANT:** This setup uses `python train.py`, NOT `uv run train.py`. The `uv run` command will re-sync dependencies and overwrite our compatible PyTorch 2.0.1 install.

This setup runs on a GTX 1080 Ti (Pascal architecture, sm_61, 11 GB VRAM) with PyTorch 2.0.1+cu118. Several features available on modern GPUs are absent. Violating any of the following constraints will crash the run:

### Hard constraints — do not violate:

- **No `torch.compile`** — disabled via commented-out decorators. Do not re-enable. PyTorch 2.0.1 Dynamo is too buggy on Pascal.
- **No bfloat16** — Pascal does not support bf16. All dtype references use `torch.float16` / `.half()`.
- **No Flash Attention 3** — FA3 requires Hopper (H100). Attention uses `sdpa_attn_func()`, a wrapper around `F.scaled_dot_product_attention`, already in train.py.
- **No `F.rms_norm`** — did not exist in PyTorch 2.0.1. A manual `norm()` function is already in train.py.
- **No `.lerp_()` with mixed dtypes** — PyTorch 2.0.1 requires matching tensor dtypes for `lerp_`. Use `.mul_()` + `.add_()` with `.item()` on scalar tensors instead.
- **No `torch._foreach_copy_`** or other `_foreach_*` ops — do not exist in 2.0.1. Use plain Python loops.
- **No `enable_gqa=` kwarg in SDPA** — not available in PyTorch 2.0.1. For GQA, manually repeat k/v heads (already handled in `sdpa_attn_func`).
- **Optimizer scalar tensors** are on `device="cuda"` with `dtype=torch.float32`. Do not move them to CPU.

### Memory budget:

- 11 GB VRAM total, target ~9-10 GB usage max (leave headroom).
- Current baseline uses only 1.7 GB — there is massive room to scale up.
- The logits tensor (`batch × seq_len × vocab_size × 4 bytes`) is often the OOM bottleneck. Keep this in mind when scaling batch size or sequence length.
- If you OOM, reduce `DEVICE_BATCH_SIZE` first, then `MAX_SEQ_LEN`.

### Baseline on this platform:

| Config | Params | Tokens | Steps | VRAM | val_bpb |
|--------|--------|--------|-------|------|---------|
| depth=4, embd=256, seq=512, batch=8 | 11.5M | 20.3M | 309 | 1.7GB | **1.344** |
| depth=6, embd=384, seq=512, batch=8 | 26.3M | 11.5M | 351 | 1.9GB | 1.348 |
| depth=8, embd=512, seq=1024, batch=8 | 50.3M | 6.9M | 105 | 5.8GB | 1.593 |

Key insight: on this hardware with a 5-minute budget, token throughput competes with model size. The small model sees 2x more data and edges out the mid model. The large model is badly undertrained. The sweet spot is somewhere in the 15-30M param range.

### Performance notes:

- ~65K tokens/sec at depth=4, ~20K tokens/sec at depth=8.
- WINDOW_PATTERN is set to "L" (full context). SDPA does not support efficient sliding window, so do not use "SSSL" or other patterns with "S".
- No tensor cores on Pascal. fp16 gives modest speedup over fp32 but not the 2x of Volta+.

### Reference:

- Parent project with wider platform support: https://github.com/karpathy/nanochat
- nanochat journey and design decisions: https://github.com/karpathy/nanochat/discussions/481
