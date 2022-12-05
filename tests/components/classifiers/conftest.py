from pytest import fixture

from edspdf.models import PDFDoc, TextBox


@fixture
def single_page_doc() -> PDFDoc:
    pdf_doc = PDFDoc(
        id="doc",
        content=b"",
        lines=[
            TextBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2, page_width=1.0, page_height=1.0),
            TextBox(x0=0.1, y0=0.6, x1=0.4, y1=0.7, page_width=1.0, page_height=1.0),
            TextBox(x0=0.1, y0=0.6, x1=0.9, y1=0.7, page_width=1.0, page_height=1.0),
        ],
    )

    return pdf_doc


@fixture
def multi_page_doc() -> PDFDoc:
    pdf_doc = PDFDoc(
        id="doc",
        content=b"",
        lines=[
            TextBox(
                x0=0.1, y0=0.1, x1=0.9, y1=0.2, page=0, page_width=1.0, page_height=1.0
            ),
            TextBox(
                x0=0.1, y0=0.6, x1=0.4, y1=0.7, page=0, page_width=1.0, page_height=1.0
            ),
            TextBox(
                x0=0.1, y0=0.6, x1=0.9, y1=0.7, page=0, page_width=1.0, page_height=1.0
            ),
            TextBox(
                x0=0.1, y0=0.1, x1=0.9, y1=0.2, page=1, page_width=1.0, page_height=1.0
            ),
            TextBox(
                x0=0.1, y0=0.6, x1=0.4, y1=0.7, page=1, page_width=1.0, page_height=1.0
            ),
            TextBox(
                x0=0.1, y0=0.6, x1=0.9, y1=0.7, page=1, page_width=1.0, page_height=1.0
            ),
        ],
    )

    return pdf_doc
