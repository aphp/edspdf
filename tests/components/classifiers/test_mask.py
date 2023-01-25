from edspdf import Component, Config
from edspdf.models import PDFDoc

configuration = """
[classifier]
@factory = "mask-classifier"
x0 = 0
y0 = 0.5
x1 = 0.5
y1 = 1
threshold = 0.4
"""

configuration_custom = """
[classifier]
@factory = "multi-mask-classifier"
threshold = 0.9

[classifier.body]
label = "body"
x0 = 0
y0 = 0.5
x1 = 0.5
y1 = 1
"""


# def test_create_mask():
#
#     with raises(ValueError):
#         Mask(label="", x0=1, x1=0)
#
#     with raises(ValueError):
#         Mask(label="", x0=0.5, x1=0.5)
#
#     with raises(ValueError):
#         Mask(label="", y0=0.5, y1=0.5)
#
#     with raises(ValueError):
#         Mask(label="", y1=0)
#
#     with raises(ValueError):
#         Mask(label="", y0=1, y1=0)
#
#     Mask(label="")


def test_simple_mask(single_page_doc):
    classifier = Config.from_str(configuration).resolve()["classifier"]

    single_page_doc = classifier(single_page_doc)

    p1, p2, p3 = [b.label for b in single_page_doc.lines]

    assert p1 == "pollution"
    assert p2 == "body"
    assert p3 == "body"


def test_custom_mask(single_page_doc):

    classifier = Config.from_str(configuration_custom).resolve()["classifier"]

    single_page_doc = classifier(single_page_doc)

    p1, p2, p3 = [b.label for b in single_page_doc.lines]

    assert p1 == "pollution"
    assert p2 == "body"
    assert p3 == "pollution"
