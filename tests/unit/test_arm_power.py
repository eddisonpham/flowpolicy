"""Size of the paired A/B: the closed form and the exact test it is standing in for.

`pairwise_binary_n` is a closed form, but the test that would actually run is the exact McNemar, so
a sizing number that the exact test does not honour is worse than no number. These tests pin three
things: the formula reproduces the figure `docs/10` reached by reasoning (`~600` for 8 pp at the
44 % discordance the sampler comparison showed), the simulated power at that n is near the nominal
0.8, and the estimator moves in the directions it must (more episodes for smaller effects, for
higher discordance).

No result in `results/` depends on this. It exists so `docs/10` section 5 can quote a size that is
computed rather than asserted.
"""

from __future__ import annotations

import pytest

from policybench.eval.stats import mcnemar_power, paired_binary_n, resolvable_effect

pytestmark = pytest.mark.unit


def test_reproduces_the_600_seed_figure_the_doc_reasoned_to() -> None:
    # docs/10 sized 600 by hand for 8 pp at psi = 0.44; the closed form lands within ~11 % of it,
    # which is the agreement that matters -- the hand figure was a rounded rule of thumb.
    assert 500 <= paired_binary_n(0.08, 0.44) <= 620


def test_a_4pp_effect_needs_more_episodes_than_8pp() -> None:
    assert paired_binary_n(0.04, 0.20) > paired_binary_n(0.08, 0.20)


def test_more_discordance_needs_more_episodes() -> None:
    sizes = [paired_binary_n(0.04, psi) for psi in (0.05, 0.10, 0.20, 0.30, 0.44)]
    assert sizes == sorted(sizes)
    # The range is what makes a single prior unsafe: 5 % discordance is ~9x cheaper than 44 %.
    assert sizes[-1] > 5 * sizes[0]


def test_simulated_power_matches_the_nominal_80_percent() -> None:
    n = paired_binary_n(0.04, 0.44)
    assert mcnemar_power(n, 0.04, 0.44, trials=600, seed=0) == pytest.approx(0.8, abs=0.07)


def test_power_increases_with_n() -> None:
    small = mcnemar_power(300, 0.04, 0.44, trials=400, seed=1)
    large = mcnemar_power(2250, 0.04, 0.44, trials=400, seed=1)
    assert small < large
    assert large > 0.75


def test_a_zero_effect_or_zero_discordance_is_an_error_not_a_number() -> None:
    with pytest.raises(ValueError):
        paired_binary_n(0.0, 0.2)
    with pytest.raises(ValueError):
        paired_binary_n(0.04, 0.0)


def test_resolvable_effect_inverts_the_size_formula() -> None:
    # Whatever n the formula asks for, asking the inverse for that n must return the same effect.
    for effect, psi in ((0.05, 0.20), (0.04, 0.44), (0.08, 0.10)):
        n = paired_binary_n(effect, psi)
        assert resolvable_effect(n, psi) == pytest.approx(effect, abs=0.004)


def test_a_200_episode_block_resolves_about_13_points() -> None:
    # The number the audit turns on: every null in results/ sits on a block this coarse.
    assert resolvable_effect(200, 0.44) == pytest.approx(0.13, abs=0.02)
    assert resolvable_effect(2200, 0.44) < 0.05


def test_resolvable_effect_rejects_an_empty_block() -> None:
    with pytest.raises(ValueError):
        resolvable_effect(0, 0.2)
