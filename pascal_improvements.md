# train.py Improvements for Pascal — Informed by nanochat

## Already done (patches applied during setup):
1. FA3 → sdpa_attn_func() with SDPA + manual sliding window mask
2. bfloat16 → float16 everywhere
3. F.rms_norm → manual x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
4. torch.compile disabled
5. lerp_ → mul_ + add_ with .item()
6. _foreach_copy_ → plain loop
7. Optimizer scalar tensors moved to CUDA
8. MAX_SEQ_LEN overridden to 512

## Recommended improvements (agent or human can apply):

### 1. Improve the SDPA attention wrapper
Current sdpa_attn_func() builds an explicit attention mask for sliding window,
which allocates a T×T float tensor on every forward pass. Since we use
WINDOW_PATTERN="L" (full context, no sliding window), this code path is never
hit, but it's still checked. A cleaner version from nanochat's flash_attention.py:

```python
def sdpa_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
    # (B, T, H, D) -> (B, H, T, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    # GQA head expansion
    if k.size(1) != q.size(1):
        reps = q.size(1) // k.size(1)
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return y.transpose(1, 2)
```

Since WINDOW_PATTERN="L", we never need sliding window masking. This is simpler
and avoids accidental T×T mask allocation if the agent experiments with larger
sequence lengths. If the agent wants to try sliding window patterns, it would
need to add the mask back.

### 2. Gradient checkpointing for larger models
PyTorch 2.0.1 supports torch.utils.checkpoint. For depth=8+ models that OOM
during backward, wrapping each transformer block in gradient checkpointing
trades compute for VRAM — roughly halves peak memory at the cost of ~30% slower
training. This could allow depth=8 to fit where it currently OOMs.

```python
from torch.utils.checkpoint import checkpoint

# In the forward loop over blocks:
for i, block in enumerate(self.transformer.blocks):
    x = checkpoint(block, x, ...)  # recompute activations during backward
```

### 3. Value embedding gate scale
nanochat uses `3 * sigmoid(...)` for the value embedding gate, while autoresearch
uses `2 * sigmoid(...)`. The nanochat version gives the gate a range of (0, 3)
instead of (0, 2), allowing stronger value residual mixing. This is a one-line
change the agent can try.

### 4. Mixed precision scaler
We use torch.amp.autocast with float16, but we're not using a GradScaler.
For fp16 training, GradScaler helps avoid underflow in gradients:

```python
scaler = torch.cuda.amp.GradScaler()
# In training loop:
with autocast_ctx:
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

This could improve training stability, especially for deeper models where
gradients get small. The agent should try this if it sees NaN losses or
unstable training.

### 5. Compute-optimal batch size
The current TOTAL_BATCH_SIZE=2**16 (65K tokens) with DEVICE_BATCH_SIZE=8 and
seq_len=512 gives grad_accum_steps=16. That means 16 forward/backward passes
per optimizer step, but only 1 step every 16 seconds.

Try TOTAL_BATCH_SIZE=2**14 (16K) for grad_accum=4, giving 4x more optimizer
updates in the same wall time. Smaller total batch with more steps often beats
larger batch with fewer steps at small model scales.

### 6. Peak FLOPS reference
The MFU calculation in train.py uses H100_BF16_PEAK_FLOPS as reference. This
is cosmetic (doesn't affect training), but the agent could update it to the
actual 1080 Ti peak FLOPS (~11.3 TFLOPS fp32, ~22.6 TFLOPS fp16 with dp4a
tricks, though Pascal doesn't have real fp16 tensor cores). This would give
a more meaningful MFU number for experiment comparison.
