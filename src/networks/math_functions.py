import numpy as np


def std(values, ave):
    return np.sqrt(float(sum((value - ave) ** 2
                          for value in values)) / len(values))


def round(a, base=4):
    """ Rounds a given number to the nearest specified base.
    """
    return int(base * round(a / base))


def return_percent(original, new):
    """ Returns a number as a percentage change from an original number."""
    if original < 0:
        new -= original
        original = 0
    return -(100 - new / float(original) * 100)
