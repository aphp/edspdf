from confit import Config

import edspdf

configuration = """
[pipeline]
pipeline = ["classifier"]
components = ${components}

[components.classifier]
@factory = "mask-classifier"
x0 = 0
y0 = 0.5
x1 = 0.5
y1 = 1
threshold = 0.4
"""

configuration_custom = """
[pipeline]
pipeline = ["classifier"]
components = ${components}

[components.classifier]
@factory = "multi-mask-classifier"
threshold = 0.9

[components.classifier.body]
label = "body"
x0 = 0
y0 = 0.5
x1 = 0.5
y1 = 1
"""


def test_simple_mask(single_page_doc):
    model = edspdf.load(Config.from_str(configuration))

    single_page_doc = model(single_page_doc)

    p1, p2, p3 = [b.label for b in single_page_doc.text_boxes]

    assert p1 == "pollution"
    assert p2 == "body"
    assert p3 == "body"


def test_custom_mask(single_page_doc):
    model = edspdf.load(Config.from_str(configuration_custom))

    single_page_doc = model(single_page_doc)

    p1, p2, p3 = [b.label for b in single_page_doc.text_boxes]

    assert p1 == "pollution"
    assert p2 == "body"
    assert p3 == "pollution"
