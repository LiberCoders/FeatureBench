import pytest
from featurebench_fixture import feature


def test_feature_zero() -> None:
    assert feature(0) == 3


def test_feature_positive() -> None:
    assert feature(4) == 11


def test_feature_rejects_negative() -> None:
    with pytest.raises(ValueError):
        feature(-1)
