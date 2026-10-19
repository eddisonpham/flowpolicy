"""G4: capture a small module loop in a CUDA graph and replay it.

Spec: docs/02_environment_setup.md section 7. Requirements noted in docs/05
section 1 (static shapes, static buffers, warmup before capture).
"""

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available, G4 needs a GPU")
        return 1

    model = torch.nn.Linear(64, 64).cuda().eval()
    static_x = torch.randn(1, 64, device="cuda")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            model(static_x)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_y = model(static_x)

    static_x.copy_(torch.randn(1, 64, device="cuda"))
    graph.replay()
    with torch.no_grad():
        expected = model(static_x)

    try:
        torch.testing.assert_close(static_y, expected, rtol=1e-5, atol=1e-5)
    except AssertionError as exc:
        print(f"FAIL: replay mismatch: {exc}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
