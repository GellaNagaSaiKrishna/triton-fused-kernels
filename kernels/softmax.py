"""Fused row-wise softmax in Triton.

A naive softmax is five kernels, each a full pass over memory:
    m = x.max(dim=1); z = x - m; e = exp(z); s = e.sum(dim=1); y = e / s
The fused kernel loads one row into on-chip memory (registers / SRAM),
does all five steps there, and writes the row back once.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)          # subtract row max for numerical stability
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension of a 2-D tensor, one Triton program per row."""
    assert x.dim() == 2, "expects a 2-D tensor"
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    assert block <= 65536, "row too long for a single block"
    y = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=block)
    return y


def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """Unfused baseline: five separate PyTorch ops."""
    m = x.max(dim=1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=1, keepdim=True)
