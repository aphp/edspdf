from edspdf.pipeline import Pipeline
from edspdf.structures import Box
from edspdf.visualization.merge import merge_boxes


def test_merge():
    lines = [
        Box(page_num=0, x0=0, x1=1, y0=0, y1=0.1, label="body"),
        Box(page_num=0, x0=0, x1=1, y0=0.1, y1=0.2, label="body"),
        Box(page_num=0, x0=0, x1=0.4, y0=0.2, y1=0.3, label="body"),
        Box(page_num=0, x0=0.6, x1=1, y0=0.2, y1=0.3, label="other"),
        Box(page_num=1, x0=0.6, x1=1, y0=0.2, y1=0.3, label="body"),
    ]

    merged = [
        Box(page_num=0, x0=0.0, x1=1.0, y0=0.0, y1=0.2, label="body"),
        Box(page_num=0, x0=0.0, x1=0.4, y0=0.2, y1=0.3, label="body"),
        Box(page_num=0, x0=0.6, x1=1.0, y0=0.2, y1=0.3, label="other"),
        Box(page_num=1, x0=0.6, x1=1.0, y0=0.2, y1=0.3, label="body"),
    ]

    out = merge_boxes(lines)

    assert len(out) == 4

    assert out == merged


def test_pipeline(pdf, blank_pdf):
    model = Pipeline()
    model.add_pipe("pdfminer-extractor")
    model.add_pipe(
        "mask-classifier", config=dict(x0=0.1, y0=0.4, x1=0.5, y1=0.9, threshold=0.1)
    )

    pdf_pages = model(pdf).pages
    blank_pdf_pages = model(blank_pdf).pages
    assert len([b for p in pdf_pages for b in merge_boxes(p.text_boxes)]) == 7
    assert len([b for p in blank_pdf_pages for b in merge_boxes(p.text_boxes)]) == 0
