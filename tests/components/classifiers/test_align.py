from edspdf.models import TextBox
from edspdf.utils.alignment import align_box_labels


def test_align_multi_page(multi_page_doc):

    annotations = [
        TextBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0, page=0, label="big"),
        TextBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9, page=1, label="small"),
    ]

    labelled = align_box_labels(annotations, multi_page_doc.lines)
    assert [b.label for b in labelled] == [
        "big",
        "big",
        "big",
        "small",
        "small",
        "small",
    ]


def test_align_cross_page(multi_page_doc):

    annotations = [
        TextBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0, label="big"),
        TextBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9, label="small"),
    ]

    labelled = align_box_labels(annotations, multi_page_doc.lines)
    assert [b.label for b in labelled] == [
        "small",
        "small",
        "small",
        "small",
        "small",
        "small",
    ]
