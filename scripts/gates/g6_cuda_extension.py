"""G6: build and run a trivial CUDA C++ extension.

Needs nvcc (docs/02 section 7). This host has no CUDA toolkit, so the gate
prints NOT RUN and records why. A red G6 removes the optional CUDA stretch (D6).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("NOT RUN: nvcc not found on PATH; D6 CUDA-stretch stays closed")
        return 0
    if not _has_compiler():
        print("NOT RUN: no host C++ compiler found")
        return 0
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:  # the build tool is part of the finding
        print(f"FAIL: cannot import torch cpp_extension: {type(exc).__name__}: {exc}")
        return 1
    source = "torch::Tensor twice(torch::Tensor x) { return x * 2.0; }"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            module = load_inline(
                name="g6_twice",
                cpp_sources="torch::Tensor twice(torch::Tensor x);",
                cuda_sources=source
                + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("twice", &twice); }',
                functions=["twice"],
                extra_cuda_cflags=["-O2"],
                build_directory=tmp,
                verbose=False,
            )
        except Exception as exc:
            print(f"FAIL: extension build/load failed: {type(exc).__name__}: {exc}")
            return 1
    import torch

    if torch.cuda.is_available():
        out = module.twice(torch.ones(4, device="cuda"))
        if not torch.allclose(out.cpu(), torch.full((4,), 2.0)):
            print("FAIL: extension returned wrong values")
            return 1
    print(f"PASS: built and loaded with {Path(nvcc).name}")
    return 0


def _has_compiler() -> bool:
    import os

    return bool(os.environ.get("CXX")) or any(
        shutil.which(c) for c in ("cl", "g++", "gcc", "clang++")
    )


if __name__ == "__main__":
    sys.exit(main())
