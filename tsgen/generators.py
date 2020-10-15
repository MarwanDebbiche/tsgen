"""
TimeSerie generators.

This module contains several generator
functions which generate TimeSerie objects
with different trends.
"""

import math

import numpy as np
import pandas as pd


from tsgen.time_serie import TimeSerie


def affine(start, end, freq, start_y, end_y):
    """
    Generate a linear TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index, y_values=np.linspace(start_y, end_y, len(index))
    )


def constant(start, end, freq, value):
    """
    Generate a constant TimeSerie.
    """
    return affine(start, end, freq, value, value)


def cosine(start, end, freq, amp=1, n_periods=1):
    """
    Generate a cosine TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index,
        y_values=amp
        * np.cos(np.linspace(0, 2 * math.pi * n_periods, num=len(index))),
    )


def sine(start, end, freq, n_periods=1):
    """
    Generate a sine TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index,
        y_values=np.sin(
            np.linspace(0, 2 * math.pi * n_periods, num=len(index))
        ),
    )


def randn(start, end, freq, mean=0, std=1):
    """
    Generate a random normally distributed TimeSerie.
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index, y_values=(std * np.random.randn(len(index)) + mean)
    )
