"""Resumable trainer (docs/03 section 5, docs/07 M3).

Determinism: the data order comes from a torch.Generator saved in the
checkpoint, and the global torch RNG state is saved too, so an interrupted run
resumes the same stream of windows and noise. Resume equivalence is tested at
`rtol=1e-5`; the docs and README say "tolerance-based", never "bit-exact".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from flowpolicy.config import TrainConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.data.window import WindowDataset
from flowpolicy.models.policies import BasePolicy, build_policy
from flowpolicy.training.ema import ModelEMA


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _lock_hash(path: str | Path = "uv.lock") -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "unknown"


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        stats: Stats,
        train_data: WindowDataset,
        val_data: WindowDataset | None = None,
        device: str | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.stats = stats
        self.train_data = train_data
        self.val_data = val_data
        self.device = torch.device(device or cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cfg.device is cuda but torch.cuda.is_available() is False")

        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)

        self.policy: BasePolicy = build_policy(
            cfg.policy.kind,
            stats,
            cfg.data.obs_horizon,
            cfg.data.pred_horizon,
            cfg.data.state_dim,
            cfg.data.action_dim,
            cfg.model,
            ddpm_train_steps=cfg.policy.ddpm_train_steps,
            flow_time_sampling=cfg.policy.flow_time_sampling,
        ).to(self.device)
        self.ema = ModelEMA(self.policy, cfg.ema_decay)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: cosine_with_warmup(s, cfg.warmup_steps, cfg.steps)
        )
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self._stop = False
        self._rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.log_path = Path(cfg.results_dir) / "train" / f"{cfg.run_id}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(cfg.out_dir) / cfg.run_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _batch(self, size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, len(self.train_data), (size,), generator=self._rng).numpy()
        batch = self.train_data.get_batch(idx)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _val_loss(self) -> float:
        if self.val_data is None or len(self.val_data) == 0:
            return float("nan")
        n = min(128, len(self.val_data))
        idx = np.linspace(0, len(self.val_data) - 1, n).astype(np.int64)
        batch = {k: v.to(self.device) for k, v in self.val_data.get_batch(idx).items()}
        self.policy.eval()
        with torch.no_grad():
            return float(self.policy.loss(batch)["loss"].item())

    def _log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def save(self, path: str | Path | None = None, *, final: bool = False) -> Path:
        path = Path(path) if path is not None else self.ckpt_dir / "last.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": self.policy.state_dict(),
            "ema": self.ema.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "config": asdict(self.cfg),
            "stats": asdict(self.stats),
            "rng": self._rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "git": _git_hash(),
            "lock_sha256": _lock_hash(),
            "policy_kind": self.cfg.policy.kind,
            "final": final,
        }
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def resume(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(payload["model"])
        self.ema.module.load_state_dict(payload["ema"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.step = int(payload["step"])
        # A checkpoint loaded with map_location=the training device holds CUDA
        # tensors; the RNG setters only accept CPU ByteTensors.
        self._rng.set_state(payload["rng"].cpu())
        torch.set_rng_state(payload["torch_rng"].cpu())
        if self.device.type == "cuda" and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            print(f"signal {signum}: saving a final checkpoint and exiting", flush=True)
            self._stop = True

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except ValueError:  # not in the main thread (tests)
            pass

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        self._install_signal_handlers()
        limit = self.cfg.steps if max_steps is None else min(self.cfg.steps, max_steps)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        last_log = start
        last_loss = float("nan")
        n_params = self.policy.num_parameters()
        print(
            json.dumps(
                {
                    "run_id": self.cfg.run_id,
                    "policy": self.cfg.policy.kind,
                    "step": self.step,
                    "params": n_params,
                }
            ),
            flush=True,
        )
        while self.step < limit and not self._stop:
            batch = self._batch(self.cfg.batch_size)
            self.policy.train()
            self.optimizer.zero_grad(set_to_none=True)
            use_bf16 = self.cfg.precision == "bf16" and self.device.type == "cuda"
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                out = self.policy.loss(batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {self.step}: {loss.item()}")
            loss.backward()  # type: ignore[no-untyped-call]
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            self.ema.update(self.policy, self.step)
            last_loss = float(loss.detach().item())

            if self.step % self.cfg.log_every == 0 or self.step == 1:
                now = time.time()
                record = {
                    "run_id": self.cfg.run_id,
                    "policy": self.cfg.policy.kind,
                    "step": self.step,
                    "loss": last_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                    "steps_per_s": self.cfg.log_every / max(1e-9, now - last_log),
                    "elapsed_s": now - start,
                }
                if self.device.type == "cuda":
                    record["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
                self._log(record)
                self.history.append(record)
                last_log = now
            if self.cfg.val_every > 0 and self.step % self.cfg.val_every == 0:
                record = {
                    "run_id": self.cfg.run_id,
                    "step": self.step,
                    "val_loss": self._val_loss(),
                }
                self._log(record)
                self.history.append(record)
            if self.cfg.ckpt_every > 0 and self.step % self.cfg.ckpt_every == 0:
                self.save(self.ckpt_dir / f"step_{self.step:07d}.pt")
        final_path = self.save(final=True)
        summary = {
            "run_id": self.cfg.run_id,
            "policy": self.cfg.policy.kind,
            "step": self.step,
            "loss": last_loss,
            "params": n_params,
            "elapsed_s": time.time() - start,
            "checkpoint": str(final_path),
            "stopped_early": self.step < self.cfg.steps,
        }
        if self.device.type == "cuda":
            summary["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        self._log({**summary, "event": "end"})
        return summary
