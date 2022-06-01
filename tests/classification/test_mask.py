import pandas as pd
from pytest import raises
from thinc.config import Config

from edspdf import registry
from edspdf.classification.mask import Mask

configuration = """
[classifier]
@classifiers = "mask.v1"
x0 = 0
y0 = 0.5
x1 = 0.5
y1 = 1
threshold = 0.9
"""

configuration_custom = """
[classifier]
@classifiers = "custom_masks.v1"

[classifier.body]
label = "body"
x0 = 0
y0 = 0.5
x1 = 0.5
y1 = 1
threshold = 0.9
"""


def test_create_mask():

    with raises(ValueError):
        Mask(label="", x0=1, x1=0)

    with raises(ValueError):
        Mask(label="", x0=0.5, x1=0.5)

    with raises(ValueError):
        Mask(label="", y0=0.5, y1=0.5)

    with raises(ValueError):
        Mask(label="", y1=0)

    with raises(ValueError):
        Mask(label="", y0=1, y1=0)

    Mask(label="")


def test_simple_mask(lines):
    classifier = registry.resolve(Config().from_str(configuration))["classifier"]

    del classifier.comparison["threshold"]

    lines["label"] = classifier.predict(lines)

    p1, p2, p3 = lines.label.values

    assert p1 == "pollution"
    assert p2 == "body"
    assert p3 == "pollution"


def test_custom_mask(lines):

    classifier = registry.resolve(Config().from_str(configuration_custom))["classifier"]

    lines["label"] = classifier.predict(lines)

    p1, p2, p3 = lines.label.values

    assert p1 == "pollution"
    assert p2 == "body"
    assert p3 == "pollution"
