from edspdf import Component, Config
from edspdf.models import PDFDoc

configuration = """
[classifier]
@factory = "dummy-classifier"
label = "body"
"""


def test_dummy(single_page_doc):
    classifier = Config.from_str(configuration).resolve()["classifier"]

    single_page_doc = classifier(single_page_doc)

    p1, p2, p3 = [b.label for b in single_page_doc.lines]

    assert p1 == "body"
    assert p2 == "body"
    assert p3 == "body"
