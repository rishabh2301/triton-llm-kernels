"""Correctness tests against plain PyTorch references.

On a GPU:     pytest -q
Without GPU:  TRITON_INTERPRET=1 pytest -q   (Triton's CPU interpreter, slow but exact)
"""

import os

import pytest
import torch

from kernels import rmsnorm, rmsnorm_reference, swiglu, swiglu_reference

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() and not INTERPRET else "cpu"
if DEVICE == "cpu" and not INTERPRET:
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

DTYPES = [torch.float32] if DEVICE == "cpu" else [torch.float32, torch.float16, torch.bfloat16]
TOL = {torch.float32: (1e-5, 1e-5), torch.float16: (1e-2, 1e-2), torch.bfloat16: (3e-2, 3e-2)}
SHAPES = [(4, 64), (3, 5, 100), (7, 1000)] if DEVICE == "cpu" else [(8, 4096), (2, 512, 3072), (33, 8192)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_forward_backward(shape, dtype):
    torch.manual_seed(0)
    atol, rtol = TOL[dtype]
    N = shape[-1]
    x = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
    w = (1 + 0.1 * torch.randn(N, device=DEVICE, dtype=dtype)).requires_grad_(True)
    dy = torch.randn(shape, device=DEVICE, dtype=dtype)

    y = rmsnorm(x, w)
    y.backward(dy)
    dx, dw = x.grad.clone(), w.grad.clone()
    x.grad, w.grad = None, None

    y_ref = rmsnorm_reference(x, w)
    y_ref.backward(dy)

    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx, x.grad, atol=atol, rtol=rtol)
    # dW sums over every row, so its absolute error grows with the row count.
    torch.testing.assert_close(dw, w.grad, atol=atol * 10, rtol=rtol * 10)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_forward_backward(shape, dtype):
    torch.manual_seed(0)
    atol, rtol = TOL[dtype]
    g = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
    u = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
    dout = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = swiglu(g, u)
    out.backward(dout)
    dg, du = g.grad.clone(), u.grad.clone()
    g.grad, u.grad = None, None

    out_ref = swiglu_reference(g, u)
    out_ref.backward(dout)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dg, g.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(du, u.grad, atol=atol, rtol=rtol)


def test_rmsnorm_gradcheck():
    """Finite-difference check of the analytic backward (fp64 not supported by all
    Triton ops, so this uses fp32 with loose tolerances)."""
    torch.manual_seed(0)
    x = torch.randn(3, 16, device=DEVICE, dtype=torch.float32, requires_grad=True)
    w = torch.randn(16, device=DEVICE, dtype=torch.float32, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: rmsnorm(a, b), (x, w), eps=1e-3, atol=1e-2, rtol=1e-2
    )
