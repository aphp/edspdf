from confit import Config

import edspdf
from edspdf.structures import PDFDoc

configuration = """
[pipeline]
pipeline = ["classifier"]
components = ${components}

[components.classifier]
@factory = "random-classifier"
labels = [ "body", "header" ]
"""


def test_random_classifier(single_page_doc: PDFDoc):
    model = edspdf.load(Config.from_str(configuration))

    single_page_doc = model(single_page_doc)

    assert set(b.label for b in single_page_doc.text_boxes) == {"body", "header"}
