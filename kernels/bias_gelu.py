"""Fused bias-add + GELU (tanh approximation) in Triton.

Unfused PyTorch runs two kernels:
    t = x + bias        # read x, write t
    y = gelu(t)         # read t, write y
That is 4 passes over memory for an op that does almost no arithmetic,
so it is memory-bound. The fused kernel reads x once and writes y once.
"""
import torch
import triton
import triton.language as tl

# sqrt(2/pi) * 2, used in: gelu_tanh(x) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3))
_TWO_SQRT_2_OVER_PI = 1.5957691216057308


@triton.jit
def _bias_gelu_kernel(x_ptr, bias_ptr, y_ptr, n_elements, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + (offsets % n_cols), mask=mask, other=0.0).to(tl.float32)
    t = x + b
    # tanh-GELU written with sigmoid: 0.5*t*(1+tanh(u)) == t*sigmoid(2u)
    u = _TWO_SQRT_2_OVER_PI * (t + 0.044715 * t * t * t)
    y = t * tl.sigmoid(u)

    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """y = gelu_tanh(x + bias), bias broadcast over the last dimension of x."""
    assert x.shape[-1] == bias.shape[0], "bias must match the last dimension of x"
    x = x.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)
    _bias_gelu_kernel[grid](x, bias, y, n, x.shape[-1], BLOCK_SIZE=block_size)
    return y


def bias_gelu_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Unfused PyTorch baseline (two kernels)."""
    return torch.nn.functional.gelu(x + bias, approximate="tanh")
