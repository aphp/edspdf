from thinc.config import Config

from edspdf import registry

configuration = """
[classifier]
@classifiers = "random.v1"
classes = [ "body", "header" ]
"""


def test_random_classifier(lines):
    classifier = registry.resolve(Config().from_str(configuration))["classifier"]

    lines["label"] = classifier.predict(lines)

    assert set(lines["label"]) == {"body", "header"}
