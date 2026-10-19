"""G2: bf16 autocast and scaled_dot_product_attention run and stay finite.

Spec: docs/02_environment_setup.md section 7.
"""

import sys

import torch
import torch.nn.functional as F


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available, G2 needs a GPU")
        return 1

    q = torch.randn(2, 8, 64, 32, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = F.scaled_dot_product_attention(q, q, q)
    print("out_dtype", out.dtype, "shape", tuple(out.shape))
    if out.shape != q.shape:
        print(f"FAIL: shape {tuple(out.shape)} != {tuple(q.shape)}")
        return 1
    if not bool(torch.isfinite(out.float()).all().item()):
        print("FAIL: non-finite values out of SDPA")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
