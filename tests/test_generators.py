from tsgen import TimeSerie
from tsgen.generators import constant, affine, cosine, sine, randn


# We accept 1e-10 errors
TOLERANCE = 10


def test_shape():
    start = "2020"
    end = "2021"

    assert len(cosine(start, end, "1M")) == 12
    # 2020 is a leap year, and 2021-01-01 is included
    assert len(affine(start, end, "1D", 0, 1)) == 367


def test_affine():
    start = "2020"
    end = "2021"
    freq = "1M"

    assert affine(start, end, freq, 0, 1).to_frame().y_values.iloc[0] == 0
    assert affine(start, end, freq, 0, 1).to_frame().y_values.iloc[-1] == 1
    assert (
        affine(start, end, freq, 0, 1) + affine(start, end, freq, 1, 0)
    ) == (affine(start, end, freq, 1, 1))


def test_constant():
    start = "2020"
    end = "2021"
    freq = "1M"
    assert (constant(start, end, freq, 1).y_values == 1).all()


def test_cosine():
    start = "2020"
    end = "2021"
    freq = "1M"
    assert cosine(start, end, freq, n_periods=1).y_values[0] == 1
    assert (
        round(cosine(start, end, freq, n_periods=0.5).y_values[-1], TOLERANCE)
        == -1
    )
    assert (
        round(cosine(start, end, freq, n_periods=0.25).y_values[-1], TOLERANCE)
        == 0
    )


def test_sine():
    start = "2020"
    end = "2021"
    freq = "1M"
    assert sine(start, end, freq, n_periods=1).y_values[0] == 0
    assert (
        round(sine(start, end, freq, n_periods=0.5).y_values[-1], TOLERANCE)
        == 0
    )
    assert (
        round(sine(start, end, freq, n_periods=0.25).y_values[-1], TOLERANCE)
        == 1
    )


def test_randn():
    start = "2020"
    end = "2021"
    freq = "1min"
    mean = 3
    std = 10
    assert randn(start, end, freq, mean=mean).y_values.mean().round() == mean
    assert (
        randn(start, end, freq, mean=mean, std=std).y_values.std().round()
        == std
    )
