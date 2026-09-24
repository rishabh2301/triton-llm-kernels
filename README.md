# Triton LLM Kernels: Fused RMSNorm and SwiGLU, Step by Step

A hands-on tutorial on writing **fused GPU kernels in [Triton](https://triton-lang.org/)** for two operations found in almost every modern LLM (LLaMA, Mistral, Qwen and others):

| Kernel | What it computes | Where it sits in a transformer block |
|---|---|---|
| **RMSNorm** | `y = x / sqrt(mean(x²) + eps) * w` | before attention and before the MLP |
| **SwiGLU** | `out = silu(gate) * up` | inside the MLP, between the up/gate and down projections |

Both kernels include a **backward pass**, so they can be used in training.

> Everything here is built from public material: the Triton docs and tutorials, and the RMSNorm ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) and GLU-variants ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)) papers.

---

## Why fuse these at all?

Both operations are **memory-bound**: they do only a handful of FLOPs per element, so runtime is set by how many times the tensor travels between HBM and the chip.

Eager PyTorch RMSNorm launches roughly six kernels (`pow`, `mean`, `add`, `rsqrt`, `mul`, `mul`), and most of them read or write the full activation. A fused kernel **reads each row once and writes it once**. The same holds for SwiGLU: eager mode materialises `silu(gate)` as an extra tensor, and the fused kernel doesn't.

So the metric to watch is **effective bandwidth (GB/s)** compared with your GPU's peak, not TFLOPs. That's what the benchmark reports.

---

## Repository layout

```
kernels/
  rmsnorm.py      # forward + backward kernels, autograd Function, nn.Module
  swiglu.py       # forward + backward kernels, autograd Function, LLaMA-style MLP
tests/
  test_kernels.py # correctness vs PyTorch references + finite-difference gradcheck
benchmarks/
  bench.py        # eager vs torch.compile vs Triton, forward+backward, GB/s
```

---

## Part 1: RMSNorm

### Forward

One Triton program handles one row (one token's hidden vector):

```python
x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
mean_sq = tl.sum(x * x, axis=0) / N
rstd = 1.0 / tl.sqrt(mean_sq + eps)
tl.store(RSTD + row, rstd)            # keep for the backward pass
tl.store(Y + ..., (x * rstd * w).to(Y.dtype.element_ty), mask=mask)
```

Three details matter:
1. **Accumulate in fp32** even when the inputs are bf16, or the mean of squares loses precision at large hidden sizes.
2. **`BLOCK_N = next_power_of_2(N)`**, with a mask for the tail, so any hidden size works (the tests use N=100 and N=1000 on purpose).
3. **Save `rstd`** (one float per row). It's tiny, and it saves recomputing the reduction in the backward pass.

### Backward: the derivation

Let `x̂ = x · rstd` and `g = w ⊙ dy`. Differentiating through the `rstd(x)` term gives

```
dx = rstd · ( g − x̂ · mean(g ⊙ x̂) )
dw = Σ_rows  dy ⊙ x̂
```

`dx` is a per-row computation, so it fuses naturally. `dw` is a **reduction across all rows**, which is the tricky part. There are two common options:
- `tl.atomic_add` into `dw`: simple, but contended and non-deterministic.
- **Two-stage reduction (used here):** launch `G` programs. Program `p` handles rows `p, p+G, p+2G, …` and keeps a running `dw` partial in registers, then writes one `[G, N]` buffer that a single `torch.sum` finishes. This is deterministic, and no atomics are needed.

```python
for row in range(pid, M, num_programs):
    ...
    dx = (wdy - x_hat * c) * rstd
    dw_acc += dy * x_hat
tl.store(DW_PARTIAL + pid * N + cols, dw_acc, mask=mask)
```

---

## Part 2: SwiGLU

The forward pass is purely elementwise:

```python
g = tl.load(G + offs, ...).to(tl.float32)
u = tl.load(U + offs, ...).to(tl.float32)
tl.store(OUT + offs, (g * tl.sigmoid(g) * u).to(...), mask=mask)
```

For the backward pass, with `σ = sigmoid(g)`:

```
d silu/dg = σ · (1 + g · (1 − σ))
dgate     = dout · up · d silu/dg
dup       = dout · silu(g)
```

We **recompute `σ` in the backward pass instead of saving it**. That's one extra `exp` per element, but one less full-size activation tensor to store. For long-context training, activation memory is the bottleneck, so this is usually the better trade-off.

`TritonSwiGLUMLP` wires the kernel into a standard `down(silu(gate(x)) * up(x))` MLP block.

---

## Usage

```python
import torch
from kernels import TritonRMSNorm, TritonSwiGLUMLP

norm = TritonRMSNorm(4096).cuda().bfloat16()
mlp  = TritonSwiGLUMLP(4096, 14336).cuda().bfloat16()

x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = mlp(norm(x))
y.sum().backward()
```

---

## Testing

```bash
pip install -r requirements.txt

pytest -q                      # on a CUDA GPU: fp32 / fp16 / bf16, LLM-sized shapes
TRITON_INTERPRET=1 pytest -q   # no GPU needed: Triton's CPU interpreter, small shapes
```

The tests check forward outputs and gradients against PyTorch references, and run `torch.autograd.gradcheck` (finite differences) on the RMSNorm backward.

**Verification status:** all 7 tests pass under Triton's CPU interpreter (`TRITON_INTERPRET=1`), which exercises the same kernel code paths in fp32 at small shapes. The fp16/bf16 cases and the benchmarks require a CUDA GPU.

---

## Benchmarks

Requires a CUDA GPU.

```bash
python benchmarks/bench.py --dtype bfloat16
```

This times the forward and backward pass of three versions of each operation:

- **plain PyTorch**, which runs each small step as a separate GPU operation
- **`torch.compile`**, PyTorch's built-in compiler, which merges those steps automatically
- **the Triton kernels** from this repo, which merge them by hand

It uses 4,096 tokens and layer sizes typical of real LLMs, prints a table of run times and memory bandwidth (GB/s), and saves it to `benchmarks/results.md`.

**What to expect:** the Triton kernels should be clearly faster than plain PyTorch, because they read and write GPU memory fewer times. `torch.compile` does the same kind of merging, so it is the fairer comparison.

---

## Where to go next

- Fuse the **residual add into RMSNorm** (`y = norm(x + residual)`, returning both). Most production stacks do this.
- Fuse **SwiGLU into the down-projection GEMM's prologue**, so the intermediate tensor never exists.
- Add **autotuning** (`@triton.autotune`) over `num_warps` and block sizes per GPU.
- Try an **FP8 output** for the activation, feeding an FP8 GEMM.

## References

- Triton tutorials: [fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html), [layer norm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019
- Shazeer, *GLU Variants Improve Transformer*, 2020

## License

MIT
