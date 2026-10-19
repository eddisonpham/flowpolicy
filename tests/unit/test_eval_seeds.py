import pytest

from flowpolicy.eval.metrics import assert_seed_disjoint

pytestmark = pytest.mark.unit


def test_overlapping_eval_and_train_seeds_raise():
    with pytest.raises(ValueError, match="overlap"):
        assert_seed_disjoint([0, 1, 2], [1, 10000])


def test_disjoint_seeds_pass():
    assert_seed_disjoint([0, 1, 2], [10000, 10001, 10002])
