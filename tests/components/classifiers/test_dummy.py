from edspdf import Component, Config
from edspdf.models import PDFDoc

configuration = """
[classifier]
@factory = "dummy-classifier"
label = "body"
"""


def test_dummy(single_page_doc):
    classifier: Component[PDFDoc, PDFDoc] = Config().from_str(configuration)[
        "classifier"
    ]

    single_page_doc = classifier(single_page_doc)

    p1, p2, p3 = [b.label for b in single_page_doc.lines]

    assert p1 == "body"
    assert p2 == "body"
    assert p3 == "body"
