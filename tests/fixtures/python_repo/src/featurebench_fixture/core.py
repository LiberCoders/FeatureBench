def feature(value: int) -> int:
    """Return a deterministic feature score for a non-negative input."""
    if value < 0:
        raise ValueError("value must be non-negative")
    doubled = value * 2
    bias = 3
    return doubled + bias


def stable(value: int) -> int:
    """An unaffected sentinel behavior."""
    return value + 10
