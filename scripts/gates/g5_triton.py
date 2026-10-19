"""G5: Triton compiles and runs a trivial kernel.

Spec: docs/02_environment_setup.md section 7. Pitfall 2 applies: cu13 PyTorch
wheels no longer bundle ptxas, so an assembler failure here is a real result
and points at TRITON_PTXAS_PATH, not at the kernel.
"""

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available, G5 needs a GPU")
        return 1

    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # a missing or broken triton is the gate's finding
        print(f"FAIL: cannot import triton: {type(exc).__name__}: {exc}")
        return 1

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(x_ptr + offs, mask=mask)
        b = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    try:
        add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    except Exception as exc:  # an assembler failure is a result to report, not a crash
        print(f"FAIL: kernel launch/compile failed: {type(exc).__name__}: {exc}")
        return 1

    try:
        torch.testing.assert_close(out, x + y)
    except AssertionError as exc:
        print(f"FAIL: wrong result: {exc}")
        return 1

    print("triton", triton.__version__, "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
