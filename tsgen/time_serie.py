"""TimeSerie Object."""

from __future__ import annotations

from typing import Union, Any

import pandas as pd  # type: ignore
import numpy as np  # type: ignore


Numeric = Union[int, float]


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

    def __init__(self, index: pd.DatetimeIndex, y_values):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("index should be a pandas.DatetimeIndex")

        self.index: pd.DatetimeIndex = index
        self.y_values: np.ndarray = np.array(y_values)

        if len(index) != len(y_values):
            raise ValueError("index and y_values's shapes do not match")

    def to_frame(self) -> pd.DataFrame:
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
        return self.to_frame().y_values.plot()

    def __len__(self) -> int:
        return len(self.index)

    def __str__(self) -> str:
        return str(self.to_frame())

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TimeSerie):
            return False

        return (self.index == other.index).all() and (
            self.y_values == other.y_values
        ).all()

    def __mul__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        if not self._check_operator_input(other):
            raise TypeError(
                "unsupported operand type(s) for *: 'TimeSerie' and '{}'".format(
                    type(other)
                )
            )
        if isinstance(other, TimeSerie):
            self._check_indexes_match(other)

        if isinstance(other, TimeSerie):
            return TimeSerie(
                index=self.index, y_values=(self.y_values * other.y_values)
            )

        return TimeSerie(index=self.index, y_values=(self.y_values * other))

    def __rmul__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        if not isinstance(other, TimeSerie):
            return self * (1 / other)

        inverse_other: TimeSerie = TimeSerie(
            index=self.index, y_values=(1 / other.y_values)
        )

        return self * inverse_other

    def __rtruediv__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        inverse_self: TimeSerie = TimeSerie(
            index=self.index, y_values=(1 / self.y_values)
        )

        return other * inverse_self

    def __add__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        if not self._check_operator_input(other):
            raise TypeError(
                "unsupported operand type(s) for +: 'TimeSerie' and '{}'".format(
                    type(other)
                )
            )
        if isinstance(other, TimeSerie):
            self._check_indexes_match(other)

        if isinstance(other, TimeSerie):
            return TimeSerie(
                index=self.index, y_values=(self.y_values + other.y_values)
            )

        return TimeSerie(index=self.index, y_values=(self.y_values + other))

    def __radd__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        return self.__add__(other)

    def __sub__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        if not isinstance(other, TimeSerie):
            return self + (-1 * other)

        negative_other = TimeSerie(
            index=self.index, y_values=(-1 * other.y_values)
        )

        return self + negative_other

    def __rsub__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        negative_self: TimeSerie = -1 * self

        return negative_self + other

    def __pow__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        if not self._check_operator_input(
            other, allowed_types=[int], time_serie_allowed=False
        ):
            raise TypeError(
                "unsupported operand type(s) for **: 'TimeSerie' and '{}'".format(
                    type(other)
                )
            )
        if isinstance(other, TimeSerie):
            self._check_indexes_match(other)

        if isinstance(other, TimeSerie):
            return TimeSerie(
                index=self.index, y_values=(self.y_values ** other.y_values)
            )

        return TimeSerie(index=self.index, y_values=(self.y_values ** other))

    def __rpow__(self, other: Union[Numeric, TimeSerie]) -> TimeSerie:
        if not self._check_operator_input(
            other, allowed_types=[int], time_serie_allowed=False
        ):
            raise TypeError(
                "unsupported operand type(s) for **: '{}' and 'TimeSerie'".format(
                    type(other)
                )
            )
        if isinstance(other, TimeSerie):
            self._check_indexes_match(other)

        if isinstance(other, TimeSerie):
            return TimeSerie(
                index=self.index, y_values=(other.y_values ** self.y_values)
            )

        return TimeSerie(index=self.index, y_values=(other ** self.y_values))

    @staticmethod
    def _check_operator_input(
        input_: Any, allowed_types=(int, float), time_serie_allowed=True
    ) -> bool:
        for allowed_type in allowed_types:
            if isinstance(input_, allowed_type):
                return True

        if time_serie_allowed and isinstance(input_, TimeSerie):
            return True

        return False

    def _check_indexes_match(self, other: TimeSerie) -> None:
        if (
            isinstance(other, TimeSerie)
            and not (self.index == other.index).all()
        ):
            raise ValueError("Indexes do not match")
