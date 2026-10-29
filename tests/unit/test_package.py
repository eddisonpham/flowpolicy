"""Scaffold-level checks: the package imports and reports a version.

No behavioural claim about a policy is made here; the tests in docs/03 section 6
arrive with the code they cover.
"""

import pytest

import policybench

pytestmark = pytest.mark.unit


def test_version_is_a_release_string() -> None:
    assert isinstance(policybench.__version__, str)
    assert policybench.__version__.count(".") >= 1


def test_version_matches_pyproject() -> None:
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    assert pyproject["project"]["version"] == policybench.__version__
