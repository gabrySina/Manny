import pytest

from stats import median, mode, value_range


def test_median_odd_unsorted():
    assert median([5, 1, 3]) == 3


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_mode_tie_returns_smallest():
    assert mode([3, 1, 3, 1, 2]) == 1


def test_mode_empty():
    with pytest.raises(ValueError):
        mode([])


def test_value_range():
    assert value_range([4, 9, 1]) == 8
