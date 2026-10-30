"""Make `lerobot/diffusion_pusht` loadable by a 2026 lerobot, without losing its behaviour.

The checkpoint is a 2024 artifact: it kept its eight normalization buffers
(`normalize_inputs.buffer_*`, `normalize_targets.buffer_*`, `unnormalize_outputs.buffer_*`)
inside `model.safetensors` and ships no processor files, so `lerobot-eval` refuses to load it
(`ProcessorMigrationError: policy_preprocessor.json not found`).

lerobot ships the right tool for this --
`lerobot.processor.migrate_policy_normalization` -- and it does the hard part correctly: it
extracts the statistics, rebuilds `policy_preprocessor.json` / `policy_postprocessor.json` via
`make_pre_post_processors`, and reports that the cleaned weights match the policy exactly. It
then dies saving the model, inside draccus' *encoder*:

    Exception: Couldn't encode 84

raised while serialising this checkpoint's config (`crop_shape: [84, 84]`). That leaves a
0-byte `config.json` and no `model.safetensors`, so the directory is unusable.

This script runs lerobot's migration, tolerates that known failure, and finishes the two writes
it did not reach, changing nothing else:

* `config.json` -- the checkpoint's own config with `normalization_mapping` removed, which is
  the exact field set `make_policy_config` already accepted in the migration run. Everything
  that defines the policy (crop shape, DDPM/100, `n_action_steps=8`, `n_obs_steps=2`) stays.
* `model.safetensors` -- the original weights with the eight normalization keys removed, which
  is precisely what the migration computes before it tries to save.

Then it loads the result back through `DiffusionPolicy.from_pretrained`, so a directory that
cannot actually be evaluated fails here instead of halfway through the control run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

REPO = "lerobot/diffusion_pusht"
DEFAULT_OUT = Path("checkpoints/diffusion_pusht_migrated")
NORM_KEYS = (
    "normalize_inputs.buffer_",
    "normalize_targets.buffer_",
    "unnormalize_outputs.buffer_",
)


def strip_normalization(state_dict: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint's weights without the eight normalization buffers.

    Those buffers are what `migrate_policy_normalization` moves into the processor pipelines,
    and the migration computes exactly this set before its save step dies. Kept as a function
    because it is the one transformation a reviewer has to check: everything else here is
    copying files. `tests/unit/test_lerobot_migrate.py` pins it.
    """
    return {key: value for key, value in state_dict.items() if not any(k in key for k in NORM_KEYS)}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)

    # 1. lerobot's own migration: processors first. It fails at the model save, which is why
    #    the result is checked immediately below rather than trusted.
    from lerobot.processor import migrate_policy_normalization as migration

    argv = sys.argv
    sys.argv = ["migrate_policy_normalization", "--pretrained-path", REPO, "--output-dir", str(out)]
    try:
        migration.main()
    except Exception as exc:  # the draccus encode failure is expected here
        print(f"\n[lerobot] migration stopped in its save step: {type(exc).__name__}: {exc}")
    finally:
        sys.argv = argv

    # 2. Finish the two artifacts it leaves empty or missing.
    config_path = out / "config.json"
    config = json.loads(Path(hf_hub_download(REPO, "config.json")).read_text(encoding="utf-8"))
    config.pop("normalization_mapping", None)  # now carried by the processor pipelines
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"wrote {config_path} ({config_path.stat().st_size} bytes, normalization_mapping moved out)"
    )

    state_dict = load_file(hf_hub_download(REPO, "model.safetensors"))
    clean = strip_normalization(state_dict)
    dropped = len(state_dict) - len(clean)
    save_file(clean, str(out / "model.safetensors"))
    print(
        f"wrote {out / 'model.safetensors'} ({len(clean)} tensors, "
        f"{dropped} normalization keys removed)"
    )

    # 3. Gate: the directory has to load through lerobot's normal path, or the control is void.
    missing = [
        name
        for name in (
            "model.safetensors",
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        if not (out / name).is_file() or (out / name).stat().st_size == 0
    ]
    if missing:
        raise SystemExit(f"migrated directory incomplete: {missing}")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class

    config = PreTrainedConfig.from_pretrained(out)
    policy = get_policy_class(config.type).from_pretrained(out)
    print(
        f"verified: {config.type} policy loads from {out} "
        f"({sum(p.numel() for p in policy.parameters()) / 1e6:.1f} M params, "
        f"n_action_steps={config.n_action_steps}, horizon={config.horizon})"
    )


if __name__ == "__main__":
    main()
