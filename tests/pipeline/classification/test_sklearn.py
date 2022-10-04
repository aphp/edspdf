import pandas as pd
from joblib import dump
from pytest import fixture, raises
from sklearn.dummy import DummyClassifier
from thinc.config import Config

from edspdf import registry

configuration = """
[classifier]
@classifiers = "sklearn-pipeline.v1"
path = {path}
"""


@fixture
def pipeline(tmp_path):

    path = tmp_path / "pipeline.joblib"

    clf = DummyClassifier(strategy="constant", constant="body")
    clf.fit(None, ["body"])
    dump(clf, path)

    return path


@fixture
def config(pipeline):
    text = configuration.format(path=pipeline)
    return text


def test_dummy(lines, config):
    classifier = registry.resolve(Config().from_str(config))["classifier"]

    lines["label"] = classifier.predict(lines)

    p1, p2, p3 = lines.label.values

    assert p1 == "body"
    assert p2 == "body"
    assert p3 == "body"
