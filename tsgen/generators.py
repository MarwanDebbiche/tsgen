"""
TimeSerie generators.

This module contains several generator
functions which generate TimeSerie objects
with different trends.
"""

import math
from typing import Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


from tsgen.time_serie import TimeSerie

Numeric = Union[int, float]


def affine(start, end, freq, start_y: Numeric, end_y: Numeric) -> TimeSerie:
    """
    Generate a linear TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index, y_values=np.linspace(start_y, end_y, len(index))
    )


def constant(start, end, freq, value: Numeric) -> TimeSerie:
    """
    Generate a constant TimeSerie.
    """
    return affine(start, end, freq, value, value)


def cosine(
    start, end, freq, amp: Numeric = 1, n_periods: Numeric = 1
) -> TimeSerie:
    """
    Generate a cosine TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index,
        y_values=np.cos(
            np.linspace(0, 2 * math.pi * n_periods, num=len(index))
        )
        * amp,
    )


def sine(
    start, end, freq, amp: Numeric = 1, n_periods: Numeric = 1
) -> TimeSerie:
    """
    Generate a sine TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index,
        y_values=np.sin(
            np.linspace(0, 2 * math.pi * n_periods, num=len(index))
        )
        * amp,
    )


def randn(start, end, freq, mean: Numeric = 0, std: Numeric = 1) -> TimeSerie:
    """
    Generate a random normally distributed TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index, y_values=(std * np.random.randn(len(index)) + mean)
    )
