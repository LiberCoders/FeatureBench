import pytest
from featurebench_fixture import stable


@pytest.mark.parametrize(
    ("value", "expected"), [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16)]
)
def test_stable(value: int, expected: int) -> None:
    assert stable(value) == expected
