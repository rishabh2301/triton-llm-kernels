"""Fused SwiGLU activation in Triton, forward and backward.

    out = silu(gate) * up,   silu(g) = g * sigmoid(g)

In an LLM MLP block this sits between the gate/up projections and the down
projection. Eager PyTorch materialises silu(gate) as a separate tensor and
launches two kernels; the fused kernel reads gate and up once and writes
out once. The backward pass recomputes sigmoid(gate) instead of storing it,
trading a few FLOPs for memory, which is the right trade on modern GPUs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(G, U, OUT, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    out = g * tl.sigmoid(g) * u
    tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(DOUT, G, U, DG, DU, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    dout = tl.load(DOUT + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    silu = g * sig
    # d silu / dg = sig * (1 + g * (1 - sig))
    dsilu = sig * (1.0 + g * (1.0 - sig))

    tl.store(DG + offs, (dout * u * dsilu).to(DG.dtype.element_ty), mask=mask)
    tl.store(DU + offs, (dout * silu).to(DU.dtype.element_ty), mask=mask)


BLOCK = 1024


class TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.shape == up.shape, "gate and up must have the same shape"
        gate_c, up_c = gate.contiguous(), up.contiguous()
        out = torch.empty_like(gate_c)
        n = gate_c.numel()
        grid = (triton.cdiv(n, BLOCK),)
        _swiglu_fwd_kernel[grid](gate_c, up_c, out, n, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(gate_c, up_c)
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dout = dout.contiguous()
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        grid = (triton.cdiv(n, BLOCK),)
        _swiglu_bwd_kernel[grid](dout, gate, up, dg, du, n, BLOCK=BLOCK, num_warps=4)
        return dg, du


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return TritonSwiGLUFunction.apply(gate, up)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


class TritonSwiGLUMLP(torch.nn.Module):
    """LLaMA/Falcon3-style MLP: down(silu(gate(x)) * up(x)) with the fused activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
