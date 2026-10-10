import pytest

from stats import median, mode, value_range


def test_median_odd_unsorted():
    assert median([9, 1, 5, 3, 7]) == 5


def test_median_even_unsorted():
    assert median([10, 1, 4, 7]) == 5.5


def test_median_two_values():
    assert median([8, 2]) == 5


def test_median_does_not_modify_input():
    data = [3, 1, 2]
    median(data)
    assert data == [3, 1, 2]


def test_median_empty():
    with pytest.raises(ValueError):
        median([])


def test_mode_clear_winner():
    assert mode([4, 4, 2, 9, 4]) == 4


def test_mode_tie_returns_smallest():
    assert mode([7, 2, 7, 2, 5, 5]) == 2


def test_mode_empty():
    with pytest.raises(ValueError):
        mode([])


def test_value_range_still_works():
    assert value_range([-2, 10, 3]) == 12
