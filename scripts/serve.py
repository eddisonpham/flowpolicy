"""Serve a policy over local HTTP: `just serve`.

Same loader as eval and deployment (`--ckpt hf:<repo_id>` or a trained checkpoint), same
interface, one process boundary around it. The contract and the rationale are in
`docs/05` section 4; the smoke test is `tests/unit/test_deploy_server.py`.
"""

from __future__ import annotations

import argparse

from policybench.deploy.server import serve
from policybench.eval.rollout import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    # Not type=Path: `hf:<repo_id>` is a pretrained checkpoint, not a path.
    parser.add_argument("--ckpt", default="hf:lerobot/diffusion_pusht")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=10, help="DDIM steps per chunk")
    parser.add_argument(
        "--scheduler", default="ddim", choices=["ddim", "ddpm"], help="pretrained only"
    )
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy, meta = load_policy(args.ckpt, device=args.device, scheduler=args.scheduler)
    serve(
        policy,
        host=args.host,
        port=args.port,
        name=str(meta["policy"]),
        device=args.device,
        steps=args.steps,
        action_horizon=args.action_horizon,
    )


if __name__ == "__main__":
    main()
