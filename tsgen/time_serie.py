"""TimeSerie Object."""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore


class TimeSerie:
    """
    Represent a one-dimensional time serie.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Time index.

    y_values : array_like
        One dimensional array_like object.

    Examples
    --------
    >>> start, end, freq = "2020", "2021", "1M"
    >>> index = pd.date_range(start=start, end=end, freq=freq)
    >>> ts = TimeSerie(
    ...     index=index,
    ...     y_values=np.linspace(1, 1, len(index))
    ... )

    >>> ts
                y_values
    2020-01-31  0.000000
    2020-02-29  0.090909
    2020-03-31  0.181818
    2020-04-30  0.272727
    2020-05-31  0.363636
    2020-06-30  0.454545
    2020-07-31  0.545455
    2020-08-31  0.636364
    2020-09-30  0.727273
    2020-10-31  0.818182
    2020-11-30  0.909091
    2020-12-31  1.000000

    >>> ts + 3 - ts
                y_values
    2020-01-31       3.0
    2020-02-29       3.0
    2020-03-31       3.0
    2020-04-30       3.0
    2020-05-31       3.0
    2020-06-30       3.0
    2020-07-31       3.0
    2020-08-31       3.0
    2020-09-30       3.0
    2020-10-31       3.0
    2020-11-30       3.0
    2020-12-31       3.0
    """

    def __init__(self, index, y_values):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("index should be a pandas.DatetimeIndex")
        self.index = index
        self.y_values = np.array(y_values)
        if len(y_values) != len(y_values):
            raise ValueError("index and y_values's shapes do not match")

    def to_frame(self):
        """
        Convert the TimeSerie to a pandas DataFrame
        """
        return pd.DataFrame({"y_values": self.y_values}, index=self.index)

    def plot(self):
        """
        Plot the TimeSerie using pandas
        `pd.Series.plot function
        <https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html>`_.
        """
        self.to_frame().y_values.plot()

    def __len__(self):
        return len(self.index)

    def __str__(self):
        return str(self.to_frame())

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, TimeSerie):
            return False

        return (self.index == other.index).all() and (
            self.y_values == other.y_values
        ).all()

    def __mul__(self, other):
        if (
            (not isinstance(other, TimeSerie))
            and (not isinstance(other, int))
            and (not isinstance(other, float))
        ):
            raise TypeError("Wrong y_values")
        if (
            isinstance(other, TimeSerie)
            and not (self.index == other.index).all()
        ):
            raise ValueError("Indexes do not match")

        if isinstance(other, TimeSerie):
            return TimeSerie(
                index=self.index, y_values=(self.y_values * other.y_values)
            )

        return TimeSerie(index=self.index, y_values=(self.y_values * other))

    def __add__(self, other):
        if (
            (not isinstance(other, TimeSerie))
            and (not isinstance(other, int))
            and (not isinstance(other, float))
        ):
            raise TypeError("Wrong y_values")
        if (
            isinstance(other, TimeSerie)
            and not (self.index == other.index).all()
        ):
            raise ValueError("Indexes do not match")

        if isinstance(other, TimeSerie):
            return TimeSerie(
                index=self.index, y_values=(self.y_values + other.y_values)
            )

        return TimeSerie(index=self.index, y_values=(self.y_values + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, TimeSerie):
            negative_other = TimeSerie(
                index=self.index, y_values=(-1 * other.y_values)
            )
        else:
            negative_other = -1 * other

        return self + negative_other
