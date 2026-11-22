"""Correctness tests.

On a CUDA GPU these run the real kernels. Without a GPU, run with
TRITON_INTERPRET=1 and they execute through Triton's CPU interpreter.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels import bias_gelu_reference, fused_bias_gelu, fused_softmax, softmax_naive  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = [torch.float32] + ([torch.float16] if DEVICE == "cuda" else [])
TOL = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float16: dict(atol=2e-3, rtol=2e-3)}


@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (128, 1024), (64, 3000), (3, 5, 77)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_gelu_matches_pytorch(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    b = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    torch.testing.assert_close(fused_bias_gelu(x, b), bias_gelu_reference(x, b), **TOL[dtype])


@pytest.mark.parametrize("shape", [(1, 1), (4, 17), (128, 1000), (32, 4096), (2, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_matches_pytorch(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype) * 5
    y = fused_softmax(x)
    torch.testing.assert_close(y, torch.softmax(x, dim=1), **TOL[dtype])
    torch.testing.assert_close(y, softmax_naive(x), **TOL[dtype])


def test_softmax_is_numerically_stable():
    # exp(1000) overflows without the max-subtraction trick
    x = torch.tensor([[1000.0, 1001.0, 1002.0]], device=DEVICE)
    y = fused_softmax(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.sum(dim=1), torch.ones(1, device=DEVICE))


def test_softmax_non_contiguous_rows():
    # a strided view: every other row of a bigger tensor
    big = torch.randn(64, 300, device=DEVICE)
    x = big[::2]
    torch.testing.assert_close(fused_softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-5)
