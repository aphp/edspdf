import pandas as pd
from pytest import fixture, raises
from thinc.config import Config

from edspdf import registry

configuration = """
[classifier]
@classifiers = "dummy.v1"
"""


def test_dummy(lines):
    classifier = registry.resolve(Config().from_str(configuration))["classifier"]

    lines["label"] = classifier(lines)

    p1, p2, p3 = lines.label.values

    assert p1 == "body"
    assert p2 == "body"
    assert p3 == "body"
