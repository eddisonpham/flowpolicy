"""G1: GPU visible, wheel has kernels for this architecture, matmul finite.

Spec: docs/02_environment_setup.md section 7. Prints facts, then PASS or FAIL.
"""

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("torch", torch.__version__, "cuda", torch.version.cuda)
        print("FAIL: torch.cuda.is_available() is False")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    free, total = torch.cuda.mem_get_info()
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device", name, "capability", cap)
    print("arch_list", arch)
    print("vram_total_GiB", round(total / 2**30, 2), "free_GiB", round(free / 2**30, 2))

    sm = f"sm_{cap[0]}{cap[1]}"
    if not any(a.startswith(sm) or a.startswith("compute_") for a in arch):
        print(f"FAIL: no kernels for {sm} in arch_list")
        return 1

    x = torch.randn(1024, 1024, device="cuda")
    if not bool(torch.isfinite((x @ x).sum()).item()):
        print("FAIL: matmul produced non-finite values")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
