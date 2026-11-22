# Fused GPU Kernels in Triton

Two small Triton kernels that show **operator fusion**: turning several memory-bound
PyTorch ops into one GPU kernel, benchmarked against eager PyTorch and `torch.compile`.

| Kernel | Unfused PyTorch | Fused Triton |
|---|---|---|
| **Bias + GELU** (`y = gelu(x + b)`) | 2 kernels, 4 passes over memory (intermediate tensor written then re-read) | 1 kernel, 2 passes (read `x`, write `y`) |
| **Row softmax** | 5 kernels (max, subtract, exp, sum, divide) | 1 kernel: each row stays in on-chip memory |

## Why fusion helps

Both ops do very little arithmetic per byte, so they are **memory-bound**: run time is set by
how many bytes move between GPU memory (HBM/GDDR) and the cores, not by FLOPs. Every extra
kernel writes an intermediate tensor and reads it back. Fusing removes those round trips, so
the right metric is **effective bandwidth** (GB/s) compared with the GPU's peak bandwidth.

- **Bias + GELU:** each Triton program handles a `BLOCK_SIZE` slice, adds the bias
  (broadcast over columns) and applies tanh-GELU in fp32 registers, written as
  `t * sigmoid(2*sqrt(2/pi)*(t + 0.044715 t^3))`.
- **Softmax:** one program per row; the row is loaded once (padded to a power of two,
  masked lanes set to `-inf`), max-subtracted for numerical stability, exponentiated,
  normalised and stored. Strided (non-contiguous) rows are supported.

## Layout

```
kernels/bias_gelu.py   fused bias + GELU kernel and PyTorch reference
kernels/softmax.py     fused row softmax kernel and naive 5-op reference
tests/test_kernels.py  correctness tests (fp32, plus fp16 on GPU; stability, strides)
bench.py               benchmark vs eager PyTorch and torch.compile -> results/
colab_benchmark.ipynb  run tests + benchmark on a free Colab T4 GPU
```

## Run

**On a CUDA GPU** (e.g. Google Colab, Runtime → T4 GPU):
```bash
pip install -r requirements.txt
pytest -q tests
python bench.py          # writes results/results.csv and results/RESULTS.md
```

**Without a GPU** the kernels still run, through Triton's CPU interpreter (correctness only, not speed):
```bash
TRITON_INTERPRET=1 pytest -q tests
```

## Results

See [results/RESULTS.md](results/RESULTS.md) after running `bench.py`.

## What I learned / next steps

- Fusion only pays off for memory-bound chains; compute-bound ops (matmul) need tiling instead.
- `torch.compile` (TorchInductor) generates Triton kernels like these automatically. Comparing
  against it shows how close a hand-written kernel gets to the compiler.
- Next: LayerNorm forward/backward, online softmax (rows longer than one block), autotuning
  `BLOCK_SIZE` / `num_warps` with `triton.autotune`.
