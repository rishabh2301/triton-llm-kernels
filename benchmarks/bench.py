"""Benchmark the fused Triton kernels against eager PyTorch and torch.compile.

Run on a CUDA GPU:
    python benchmarks/bench.py                # prints a table, writes results.md
    python benchmarks/bench.py --dtype float16

Reports forward+backward time and effective memory bandwidth (GB/s), since
both kernels are memory-bound: the right yardstick is how close they get to
the GPU's peak HBM bandwidth, not FLOPs.
"""

import argparse
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kernels import rmsnorm, rmsnorm_reference, swiglu, swiglu_reference  # noqa: E402

TOKENS = 4096  # batch * sequence length
HIDDEN_SIZES = [2048, 3072, 4096, 8192]          # RMSNorm width
INTERMEDIATE_SIZES = [8192, 11008, 14336, 23040]  # SwiGLU width


def bench_fwd_bwd(fn, inputs, grad_out):
    def step():
        for t in inputs:
            t.grad = None
        fn(*inputs).backward(grad_out)

    return triton.testing.do_bench(step, warmup=25, rep=100)  # ms


def gbps(num_bytes, ms):
    return num_bytes / (ms * 1e-3) / 1e9


def run_rmsnorm(dtype):
    rows = []
    compiled = torch.compile(rmsnorm_reference)
    for N in HIDDEN_SIZES:
        x = torch.randn(TOKENS, N, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.ones(N, device="cuda", dtype=dtype, requires_grad=True)
        dy = torch.randn_like(x)
        # fwd: read x, write y. bwd: read x, dy, write dx.  (weight traffic is negligible)
        moved = 5 * x.numel() * x.element_size()
        res = {"N": N}
        for name, fn in [("eager", rmsnorm_reference), ("compile", compiled), ("triton", rmsnorm)]:
            ms = bench_fwd_bwd(fn, (x, w), dy)
            res[name] = (ms, gbps(moved, ms))
        rows.append(res)
    return rows


def run_swiglu(dtype):
    rows = []
    compiled = torch.compile(swiglu_reference)
    for N in INTERMEDIATE_SIZES:
        g = torch.randn(TOKENS, N, device="cuda", dtype=dtype, requires_grad=True)
        u = torch.randn_like(g, requires_grad=True)
        dout = torch.randn_like(g)
        # fwd: read g, u, write out. bwd: read dout, g, u, write dg, du.
        moved = 8 * g.numel() * g.element_size()
        res = {"N": N}
        for name, fn in [("eager", swiglu_reference), ("compile", compiled), ("triton", swiglu)]:
            ms = bench_fwd_bwd(fn, (g, u), dout)
            res[name] = (ms, gbps(moved, ms))
        rows.append(res)
    return rows


def to_markdown(title, rows):
    lines = [
        f"#### {title}",
        "",
        "| width | eager ms (GB/s) | torch.compile ms (GB/s) | Triton ms (GB/s) | speed-up vs eager |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cells = [f"{r[k][0]:.3f} ({r[k][1]:.0f})" for k in ("eager", "compile", "triton")]
        lines.append(f"| {r['N']} | " + " | ".join(cells) + f" | {r['eager'][0] / r['triton'][0]:.2f}x |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    assert torch.cuda.is_available(), "benchmarks need a CUDA GPU"
    dtype = getattr(torch, args.dtype)

    header = (
        f"GPU: {torch.cuda.get_device_name()} | dtype: {args.dtype} | tokens: {TOKENS} | "
        f"torch {torch.__version__} | triton {triton.__version__}"
    )
    report = "\n\n".join([
        header,
        to_markdown("RMSNorm (forward + backward)", run_rmsnorm(dtype)),
        to_markdown("SwiGLU (forward + backward)", run_swiglu(dtype)),
    ])
    print(report)
    out = os.path.join(os.path.dirname(__file__), "results.md")
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
