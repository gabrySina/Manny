import pytest

from roman import from_roman, to_roman


def test_to_roman_basic():
    assert [to_roman(n) for n in (1, 4, 9, 14, 40, 90, 400)] == ["I", "IV", "IX", "XIV", "XL", "XC", "CD"]


def test_to_roman_large():
    assert to_roman(1994) == "MCMXCIV"
    assert to_roman(3999) == "MMMCMXCIX"


def test_from_roman_basic():
    assert from_roman("MCMXCIV") == 1994
    assert from_roman("XLII") == 42


def test_round_trip():
    assert all(from_roman(to_roman(n)) == n for n in range(1, 4000))


def test_to_roman_out_of_range():
    for bad in (0, -5, 4000):
        with pytest.raises(ValueError):
            to_roman(bad)


def test_from_roman_empty():
    with pytest.raises(ValueError):
        from_roman("")


def test_from_roman_unknown_characters():
    with pytest.raises(ValueError):
        from_roman("XIIZ")


def test_from_roman_non_canonical():
    for bad in ("IIII", "VX", "IC", "MMMM", "VV"):
        with pytest.raises(ValueError):
            from_roman(bad)
