from pytest import raises

from edspdf.extractors.style import models


def test_addition():

    s1 = models.Style(
        font="test",
        style="normal",
        size=10,
        upright=True,
        x0=0,
        x1=0.5,
        y0=0,
        y1=1,
    )

    s2 = models.Style(
        font="test",
        style="normal",
        size=10,
        upright=True,
        x0=0.6,
        x1=1,
        y0=0,
        y1=1,
    )

    s = s1 + s2

    assert s.x0 == 0
    assert s.x1 == 1


def test_error_raising():

    s1 = models.Style(
        font="test",
        style="normal",
        size=10,
        upright=True,
        x0=0,
        x1=0.5,
        y0=0,
        y1=1,
    )

    s2 = models.Style(
        font="test",
        style="italic",
        size=10,
        upright=True,
        x0=0.6,
        x1=1,
        y0=0,
        y1=1,
    )

    with raises(ValueError):
        s1 + s2
