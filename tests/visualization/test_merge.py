from edspdf.components import PdfMinerExtractor, simple_mask_classifier_factory
from edspdf.models import Box, TextBox
from edspdf.visualization.merge import merge_boxes

# fmt: off
lines = [
    TextBox(page=0, x0=0, x1=1, y0=0, y1=0.1, label="body"),
    TextBox(page=0, x0=0, x1=1, y0=0.1, y1=0.2, label="body"),
    TextBox(page=0, x0=0, x1=0.4, y0=0.2, y1=0.3, label="body"),
    TextBox(page=0, x0=0.6, x1=1, y0=0.2, y1=0.3, label="other"),
    TextBox(page=1, x0=0.6, x1=1, y0=0.2, y1=0.3, label="body"),
]
merged = [
    TextBox(page=0, x0=0.0, x1=1.0, y0=0.0, y1=0.2, page_width=None, page_height=None, label='body', source=None, styles=[], text=None),  # noqa: E501
    TextBox(page=0, x0=0.0, x1=0.4, y0=0.2, y1=0.3, page_width=None, page_height=None, label='body', source=None, styles=[], text=None),  # noqa: E501
    TextBox(page=0, x0=0.6, x1=1.0, y0=0.2, y1=0.3, page_width=None, page_height=None, label='other', source=None, styles=[], text=None),  # noqa: E501
    TextBox(page=1, x0=0.6, x1=1.0, y0=0.2, y1=0.3, page_width=None, page_height=None, label='body', source=None, styles=[], text=None),  # noqa: E501
]


# fmt: on


def test_merge():
    out = merge_boxes(lines)

    assert len(out) == 4

    assert out == merged


def test_pipeline(pdf, blank_pdf):
    extractor = PdfMinerExtractor()
    classifier = simple_mask_classifier_factory(
        x0=0.1, y0=0.4, x1=0.5, y1=0.9, threshold=0.1
    )

    doc = extractor(pdf)
    doc = classifier(doc)

    assert len(merge_boxes(doc.lines)) == 7

    doc = extractor(blank_pdf)
    doc = classifier(doc)

    assert len(merge_boxes(doc.lines)) == 0
