"""Fused RMSNorm in Triton, forward and backward.

    y = x / sqrt(mean(x**2) + eps) * w

A naive PyTorch implementation launches several kernels (square, mean, add,
rsqrt, two multiplies) and reads/writes the activation from HBM each time.
Here each row is loaded once, normalised in registers, and written once.
The backward pass is fused the same way; the gradient of the weight is
reduced in two stages (per-program partial sums, then one small torch.sum).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X, W, Y, RSTD,          # pointers
    stride_x, stride_y,     # row strides
    N,                      # hidden size
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Accumulate in fp32 even for bf16/fp16 inputs.
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(RSTD + row, rstd)  # saved for backward, avoids recomputation

    y = x * rstd * w
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    DY, X, W, RSTD, DX, DW_PARTIAL,
    stride_dy, stride_x, stride_dx,
    M, N,
    BLOCK_N: tl.constexpr,
):
    # Each program walks rows pid, pid + G, pid + 2G, ... and keeps a running
    # partial sum of dW in registers, so dW needs no atomics.
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row in range(pid, M, num_programs):
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row)

        x_hat = x * rstd
        wdy = w * dy
        # d/dx of x * rstd(x):  rstd * (g - x_hat * mean(g * x_hat)),  g = w * dy
        c = tl.sum(wdy * x_hat, axis=0) / N
        dx = (wdy - x_hat * c) * rstd
        tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)

        dw_acc += dy * x_hat

    tl.store(DW_PARTIAL + pid * N + cols, dw_acc, mask=mask)


def _block_n(n: int) -> int:
    block = triton.next_power_of_2(n)
    if block > 65536:
        raise ValueError(f"hidden size {n} too large for a single-block row kernel")
    return block


def _num_warps(block_n: int) -> int:
    return min(max(block_n // 256, 1), 16)


class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2d.shape
        y = torch.empty_like(x2d)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_N = _block_n(N)
        _rmsnorm_fwd_kernel[(M,)](
            x2d, weight, y, rstd,
            x2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N, num_warps=_num_warps(BLOCK_N),
        )
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.BLOCK_N = BLOCK_N
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, weight, rstd = ctx.saved_tensors
        M, N = x2d.shape
        dy2d = dy.reshape(M, N).contiguous()
        dx = torch.empty_like(x2d)

        if x2d.is_cuda:
            sm_count = torch.cuda.get_device_properties(x2d.device).multi_processor_count
            num_programs = min(M, sm_count * 4)
        else:  # TRITON_INTERPRET=1 on CPU
            num_programs = min(M, 8)
        dw_partial = torch.empty((num_programs, N), device=x2d.device, dtype=torch.float32)

        _rmsnorm_bwd_kernel[(num_programs,)](
            dy2d, x2d, weight, rstd, dx, dw_partial,
            dy2d.stride(0), x2d.stride(0), dx.stride(0),
            M, N,
            BLOCK_N=ctx.BLOCK_N, num_warps=_num_warps(ctx.BLOCK_N),
        )
        dw = dw_partial.sum(0).to(weight.dtype)
        return dx.reshape(dy.shape), dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonRMSNormFunction.apply(x, weight, eps)


class TritonRMSNorm(torch.nn.Module):
    """Drop-in replacement for an LLaMA/Falcon-style RMSNorm module."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Plain PyTorch reference, computed in fp32 like most LLM codebases."""
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)
