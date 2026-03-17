# autoresearch-pascal

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for **GTX 1080 Ti** (Pascal / sm_61) GPUs with PyTorch 2.0.1.

## What this fork changes

- **Flash Attention 3** replaced with `sdpa_attn_func()` using `F.scaled_dot_product_attention`
- **bfloat16** → **float16** everywhere (Pascal has no bf16 support)
- **`F.rms_norm`** replaced with manual RMS norm (not available in PyTorch 2.0.1)
- **`torch.compile`** disabled (Dynamo too buggy on Pascal with 2.0.1)
- **`.lerp_()`** replaced with `.mul_()` + `.add_()` using `.item()` (mixed-dtype lerp_ unsupported)
- **`torch._foreach_copy_`** replaced with plain Python loop
- Optimizer scalar tensors on `device="cuda"` instead of CPU
- `MAX_SEQ_LEN` overridden to 512
- Reduced defaults: `DEPTH=4`, `DEVICE_BATCH_SIZE=8`, `TOTAL_BATCH_SIZE=2**16`, `WINDOW_PATTERN="L"`

## Hardware requirements

- GTX 1080 Ti (11 GB VRAM) or similar Pascal-generation GPU
- CUDA 11.8 toolkit
- ~2 GB VRAM at baseline config; room to scale up to ~9-10 GB

## Quick start

```bash
# 1. Set up the environment (one-time, replaces uv sync)
bash setup_pascal.sh

# 2. Prepare data (one-time)
conda activate autoresearch
python prepare.py

# 3. Run training
export CUDA_VISIBLE_DEVICES=2
python train.py

# Or use the convenience wrapper:
./run.sh
```

**Important:** Use `python train.py`, NOT `uv run train.py`. The `uv run` command will re-sync dependencies and overwrite the compatible PyTorch 2.0.1 install.

## Baseline results

| Config | Params | Tokens | Steps | VRAM | val_bpb |
|--------|--------|--------|-------|------|---------|
| depth=4, embd=256, seq=512, batch=8 | 11.5M | 20.3M | 309 | 1.7GB | **1.344** |
| depth=6, embd=384, seq=512, batch=8 | 26.3M | 11.5M | 351 | 1.9GB | 1.348 |
| depth=8, embd=512, seq=1024, batch=8 | 50.3M | 6.9M | 105 | 5.8GB | 1.593 |

## Upstream

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [karpathy/nanochat](https://github.com/karpathy/nanochat)
