import pytest
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib  # type: ignore
from tsgen import TimeSerie


# We accept 1e-10 errors
TOLERANCE = 10


@pytest.fixture
def ts_monthly_constant():
    start = "2020"
    end = "2021"
    freq = "1M"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y_values=np.linspace(1, 1, len(index)))


@pytest.fixture
def ts_monthly_1():
    start = "2020"
    end = "2021"
    freq = "1M"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y_values=np.random.randn(len(index)))


@pytest.fixture
def ts_monthly_2():
    start = "2020"
    end = "2021"
    freq = "1M"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y_values=np.random.randn(len(index)))


@pytest.fixture
def ts_daily():
    start = "2020"
    end = "2021"
    freq = "1D"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y_values=np.random.randn(len(index)))


def test_constructor(ts_monthly_1):
    with pytest.raises(TypeError):
        TimeSerie("invalid index", [0])
    with pytest.raises(ValueError):
        start = "2020"
        end = "2021"
        freq = "1M"
        index = pd.date_range(start=start, end=end, freq=freq)
        y_values = range(10)
        TimeSerie(index, y_values)


def test_to_frame(ts_monthly_1):
    assert isinstance(ts_monthly_1.to_frame(), pd.DataFrame)
    assert ts_monthly_1.to_frame().shape == (len(ts_monthly_1), 1)


def test_to_series(ts_monthly_1):
    assert isinstance(ts_monthly_1.to_series(), pd.Series)
    assert ts_monthly_1.to_series().shape == (len(ts_monthly_1),)


def test_print():
    start = "2020"
    end = "2021"
    freq = "1Y"
    index = pd.date_range(start=start, end=end, freq=freq)
    y_values = [0]
    ts = TimeSerie(index, y_values)
    assert str(ts) == repr(ts)
    assert str(ts) == str(ts.to_frame())


def test_len(ts_monthly_1, ts_daily):
    assert len(ts_monthly_1) == 12
    # 2020 is a leap year and 2021-01-01 is included
    assert len(ts_daily) == 367


def test_plot(ts_monthly_1, ts_monthly_2):
    axes = ts_monthly_1.plot()

    assert isinstance(axes, matplotlib.axes.Axes)


# Operators


def test_equality(ts_monthly_1, ts_monthly_2):
    ts_monthly_1_eq = TimeSerie(
        index=ts_monthly_1.index, y_values=ts_monthly_1.y_values
    )

    assert ts_monthly_1 != 3
    assert ts_monthly_1 != ts_monthly_2
    assert ts_monthly_1_eq == ts_monthly_1


def test_addition(ts_monthly_1, ts_monthly_2, ts_monthly_constant, ts_daily):
    assert (ts_monthly_1 + ts_monthly_2) == (ts_monthly_2 + ts_monthly_1)
    assert (ts_monthly_1 + ts_monthly_constant) == (ts_monthly_1 + 1)
    assert (1 + ts_monthly_1) == (ts_monthly_1 + 1)

    check_equal_values(
        (42 + ts_monthly_1).y_values.tolist(),
        (ts_monthly_1.y_values + 42).tolist(),
    )

    check_equal_values(
        (ts_monthly_1 + 42).y_values.tolist(),
        (ts_monthly_1.y_values + 42).tolist(),
    )

    with pytest.raises(TypeError):
        ts_monthly_1 + "invalid"

    with pytest.raises(TypeError):
        "invalid" + ts_monthly_1

    with pytest.raises(ValueError):
        ts_monthly_1 + ts_daily


def test_subtraction(
    ts_monthly_1, ts_monthly_2, ts_monthly_constant, ts_daily
):
    assert ((ts_monthly_1 - ts_monthly_1).y_values == 0).all()
    assert ((ts_monthly_constant - 1).y_values == 0).all()
    assert ((1 - ts_monthly_constant).y_values == 0).all()

    check_equal_values(
        (42 - ts_monthly_1).y_values.tolist(),
        (42 - ts_monthly_1.y_values).tolist(),
    )

    check_equal_values(
        (ts_monthly_1 - 42).y_values.tolist(),
        (ts_monthly_1.y_values - 42).tolist(),
    )
    with pytest.raises(TypeError):
        ts_monthly_1 - "invalid"
    with pytest.raises(TypeError):
        "invalid" - ts_monthly_1

    with pytest.raises(ValueError):
        ts_monthly_1 - ts_daily


def test_multiplication(
    ts_monthly_1, ts_monthly_2, ts_monthly_constant, ts_daily
):
    assert (ts_monthly_1 * ts_monthly_2) == (ts_monthly_2 * ts_monthly_1)
    assert (ts_monthly_1 * 42) == (42 * ts_monthly_1)
    assert (ts_monthly_1 * 0 + 1) == (ts_monthly_constant)
    assert ((0 * ts_monthly_1).y_values == 0).all()

    check_equal_values(
        (ts_monthly_1 * 42).y_values.tolist(),
        (ts_monthly_1.y_values * 42).tolist(),
    )

    check_equal_values(
        (42 * ts_monthly_1).y_values.tolist(),
        (ts_monthly_1.y_values * 42).tolist(),
    )
    with pytest.raises(TypeError):
        ts_monthly_1 * "invalid"
    with pytest.raises(TypeError):
        "invalid" * ts_monthly_1

    with pytest.raises(ValueError):
        ts_monthly_1 * ts_daily


def test_division(ts_monthly_1, ts_monthly_2, ts_monthly_constant, ts_daily):
    assert (ts_monthly_1 / ts_monthly_2) == (1 / ts_monthly_2 * ts_monthly_1)

    check_equal_values(
        (42 / ts_monthly_1).y_values.tolist(),
        (42 / ts_monthly_1.y_values).tolist(),
    )
    check_equal_values(
        (ts_monthly_1 / 42).y_values.tolist(),
        (ts_monthly_1.y_values / 42).tolist(),
    )

    with pytest.raises(TypeError):
        ts_monthly_1 / "invalid"
    with pytest.raises(TypeError):
        "invalid" / ts_monthly_1

    with pytest.raises(ValueError):
        ts_monthly_1 / ts_daily


def test_pow(ts_monthly_1, ts_monthly_2, ts_monthly_constant, ts_daily):
    assert (ts_monthly_1 / ts_monthly_2) == (1 / ts_monthly_2 * ts_monthly_1)

    check_equal_values(
        (42 ** ts_monthly_1).y_values.tolist(),
        (42 ** ts_monthly_1.y_values).tolist(),
    )
    check_equal_values(
        (ts_monthly_1 ** 42).y_values.tolist(),
        (ts_monthly_1.y_values ** 42).tolist(),
    )

    with pytest.raises(TypeError):
        ts_monthly_1 ** "invalid"
    with pytest.raises(TypeError):
        "invalid" ** ts_monthly_1
    with pytest.raises(TypeError):
        ts_monthly_1 ** 2.3
    with pytest.raises(TypeError):
        2.3 ** ts_monthly_1
    with pytest.raises(TypeError):
        ts_monthly_1 ** ts_monthly_2


def check_equal_values(left_values, right_values, tol=TOLERANCE):
    if len(left_values) != len(right_values):
        raise ValueError("left_values and right_values or not the same length")
    for i in range(len(left_values)):
        check_equal_value(left_values[i], right_values[i], tol)


def check_equal_value(left_value, right_value, tol=TOLERANCE):
    assert (pd.isna(left_value) and pd.isna(right_value)) or (
        abs(left_value - right_value) < 10 ** -tol
    )
