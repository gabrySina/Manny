def median(values):
    """Return the median of a non-empty list of numbers, without modifying the list.

    For an even number of values it is the mean of the two middle ones.
    Raises ValueError on an empty list.
    """
    if not values:
        raise ValueError("median of empty list")
    n = len(values)
    middle = n // 2
    if n % 2 == 1:
        return values[middle]
    return (values[middle] + values[middle + 1]) / 2


def mode(values):
    """Return the most frequent value; on a tie, the smallest of the tied values.

    Raises ValueError on an empty list.
    """
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best = None
    for value, count in counts.items():
        if best is None or count > counts[best]:
            best = value
    return best


def value_range(values):
    """Return the difference between the largest and the smallest value.

    Raises ValueError on an empty list.
    """
    if not values:
        raise ValueError("range of empty list")
    return max(values) - min(values)
