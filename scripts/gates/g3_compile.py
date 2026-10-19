"""G3: torch.compile in default, reduce-overhead and max-autotune modes.

Spec: docs/02_environment_setup.md section 7. Compile time and first-call cost
are printed because docs/05 section 1 has to report them, not guess them.
"""

import sys
import time

import torch

MODES = ("default", "reduce-overhead", "max-autotune")


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available, G3 needs a GPU")
        return 1

    model = (
        torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))
        .cuda()
        .eval()
    )
    x = torch.randn(1, 64, device="cuda")
    with torch.no_grad():
        ref = model(x)

    failed = 0
    for mode in MODES:
        t0 = time.perf_counter()
        compiled = torch.compile(model, mode=mode)
        try:
            with torch.no_grad():
                for _ in range(3):
                    y = compiled(x)
            torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)
        except Exception as exc:  # a gate reports any failure; it never swallows one
            print(f"{mode} FAIL: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        print(f"{mode} PASS  wall_s={time.perf_counter() - t0:.2f}")

    if failed:
        print(f"FAIL: {failed}/{len(MODES)} compile modes failed")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
