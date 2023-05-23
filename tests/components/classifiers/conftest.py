from pytest import fixture

from edspdf.structures import Page, PDFDoc, TextBox


@fixture
def single_page_doc() -> PDFDoc:
    doc = PDFDoc(id="doc", content=b"", pages=[])
    doc.pages = [Page(doc=doc, page_num=0, width=1.0, height=1.0)]
    doc.content_boxes = [
        TextBox(doc=doc, page_num=0, text="foo", x0=0.1, y0=0.1, x1=0.9, y1=0.2),
        TextBox(doc=doc, page_num=0, text="foo", x0=0.1, y0=0.6, x1=0.4, y1=0.7),
        TextBox(doc=doc, page_num=0, text="foo", x0=0.1, y0=0.6, x1=0.9, y1=0.7),
    ]
    return doc


@fixture
def multi_page_doc() -> PDFDoc:
    doc = PDFDoc(id="doc", content=b"")
    doc.pages = [
        Page(doc=doc, page_num=0, width=1.0, height=1.0),
        Page(doc=doc, page_num=1, width=1.0, height=1.0),
    ]
    doc.content_boxes = [
        TextBox(doc=doc, page_num=0, text="foo", x0=0.1, y0=0.1, x1=0.9, y1=0.2),
        TextBox(doc=doc, page_num=0, text="foo", x0=0.1, y0=0.6, x1=0.4, y1=0.7),
        TextBox(doc=doc, page_num=0, text="foo", x0=0.1, y0=0.6, x1=0.9, y1=0.7),
        TextBox(doc=doc, page_num=1, text="foo", x0=0.1, y0=0.1, x1=0.9, y1=0.2),
        TextBox(doc=doc, page_num=1, text="foo", x0=0.1, y0=0.6, x1=0.4, y1=0.7),
        TextBox(doc=doc, page_num=1, text="foo", x0=0.1, y0=0.6, x1=0.9, y1=0.7),
    ]

    return doc
