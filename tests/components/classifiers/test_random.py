from edspdf import Component, Config
from edspdf.models import PDFDoc

configuration = """
[classifier]
@factory = "random-classifier"
labels = [ "body", "header" ]
"""


def test_random_classifier(single_page_doc: PDFDoc):
    classifier = Config.from_str(configuration).resolve()["classifier"]

    single_page_doc = classifier(single_page_doc)

    assert set(b.label for b in single_page_doc.lines) == {"body", "header"}
