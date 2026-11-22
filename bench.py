"""Benchmark the fused Triton kernels against PyTorch. Needs a CUDA GPU.

Reports time (ms) and effective memory bandwidth (GB/s). Both ops are
memory-bound, so GB/s against the GPU's peak bandwidth is the honest metric.

Usage:  python bench.py            (writes results/*.csv and results/RESULTS.md)
"""
import csv
import os

import torch
import triton
import triton.testing

from kernels import bias_gelu_reference, fused_bias_gelu, fused_softmax, softmax_naive

assert torch.cuda.is_available(), "bench.py needs a CUDA GPU (try Google Colab with a T4)"
DEV, DT = "cuda", torch.float16
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


ERRORS = []


def ms(fn, label="?"):
    """Time fn; on failure record the error and return NaN so the run continues."""
    try:
        return triton.testing.do_bench(fn, warmup=25, rep=100)
    except Exception as e:  # noqa: BLE001 - we want every failure reported, not a crash
        msg = f"{label}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        print("FAILED", msg)
        ERRORS.append(msg)
        return float("nan")


def safe_compile(fn):
    """torch.compile can fail on some GPU/driver combos; fall back to eager and say so."""
    try:
        return torch.compile(fn)
    except Exception as e:  # noqa: BLE001
        ERRORS.append(f"torch.compile unavailable: {type(e).__name__}: {e}")
        return fn


def gbps(nbytes, t_ms):
    return nbytes / (t_ms * 1e-3) / 1e9


def bench_bias_gelu():
    compiled = safe_compile(bias_gelu_reference)
    rows = []
    for n_rows in [1024, 4096, 16384, 65536]:
        n_cols = 4096
        x = torch.randn(n_rows, n_cols, device=DEV, dtype=DT)
        b = torch.randn(n_cols, device=DEV, dtype=DT)
        nbytes = 2 * x.numel() * x.element_size()  # ideal: read x once, write y once
        for name, fn in [("pytorch_unfused", lambda: bias_gelu_reference(x, b)),
                         ("torch_compile", lambda: compiled(x, b)),
                         ("triton_fused", lambda: fused_bias_gelu(x, b))]:
            t = ms(fn, f"bias_gelu {n_rows}x{n_cols} {name}")
            rows.append(dict(op="bias_gelu", shape=f"{n_rows}x{n_cols}", impl=name, ms=t, gbps=gbps(nbytes, t)))
    return rows


def bench_softmax():
    compiled = safe_compile(softmax_naive)
    rows = []
    for n_cols in [512, 1024, 2048, 4096, 8192]:
        n_rows = 8192
        x = torch.randn(n_rows, n_cols, device=DEV, dtype=DT)
        nbytes = 2 * x.numel() * x.element_size()
        for name, fn in [("pytorch_naive_5ops", lambda: softmax_naive(x)),
                         ("torch_softmax", lambda: torch.softmax(x, dim=1)),
                         ("torch_compile", lambda: compiled(x)),
                         ("triton_fused", lambda: fused_softmax(x))]:
            t = ms(fn, f"softmax {n_rows}x{n_cols} {name}")
            rows.append(dict(op="softmax", shape=f"{n_rows}x{n_cols}", impl=name, ms=t, gbps=gbps(nbytes, t)))
    return rows


def write(rows):
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gpu = torch.cuda.get_device_name()
    lines = [f"# Results on {gpu} (fp16)\n",
             "Effective bandwidth = (bytes read once + bytes written once) / time.\n"]
    for op in ["bias_gelu", "softmax"]:
        impls = list(dict.fromkeys(r["impl"] for r in rows if r["op"] == op))
        lines.append(f"\n## {op}\n")
        lines.append("| shape | " + " | ".join(f"{i} ms (GB/s)" for i in impls) + " |")
        lines.append("|---|" + "---|" * len(impls))
        for shape in dict.fromkeys(r["shape"] for r in rows if r["op"] == op):
            cells = []
            for i in impls:
                r = next(r for r in rows if r["op"] == op and r["shape"] == shape and r["impl"] == i)
                cells.append(f"{r['ms']:.3f} ({r['gbps']:.0f})")
            lines.append(f"| {shape} | " + " | ".join(cells) + " |")
    if ERRORS:
        lines.append("\n## Errors (nan above = this implementation failed)\n")
        lines += [f"- {e}" for e in ERRORS]
    with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(), "| capability", torch.cuda.get_device_capability(),
          "| torch", torch.__version__, "| triton", triton.__version__)
    write(bench_bias_gelu() + bench_softmax())
