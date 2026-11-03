"""`TrainConfig.validate`: the fields the run loop divides by or indexes with.

A config that reaches the trainer with `log_every = 0` is a `ZeroDivisionError` an hour into a
run, not a busy loop; `steps = 0` builds the model, trains it zero times and prints a summary;
`ema_decay = 1.0` freezes the EMA at its initial random weights, which would then be what every
evaluation measures. The guards cost nothing, so they are pinned here rather than discovered on
a GPU.
"""

from __future__ import annotations

import pytest

from policybench.config import TrainConfig

pytestmark = pytest.mark.unit


def test_the_defaults_validate() -> None:
    TrainConfig().validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("log_every", 0),
        ("steps", 0),
        ("batch_size", 0),
        ("ema_decay", 1.0),
        ("ema_decay", 0.0),
        ("warmup_steps", -1),
        ("val_every", -1),
    ],
)
def test_a_value_the_loop_cannot_use_is_refused(field: str, value: float) -> None:
    config = TrainConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_a_zero_step_sampler_is_refused() -> None:
    # `steps=0` in any of the samplers is a division by zero inside the sampling loop, so it is
    # rejected here where the message can name the field.
    config = TrainConfig()
    config.policy.flow_steps = 0
    with pytest.raises(ValueError, match="sampler step counts"):
        config.validate()
