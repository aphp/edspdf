from edspdf.components.aggregators.simple import SimpleAggregator
from edspdf.structures import Page, PDFDoc, Text, TextBox

doc = PDFDoc(
    content=b"",
    pages=[],
)
doc.pages = [
    Page(doc=doc, page_num=0, width=1, height=1),
    Page(doc=doc, page_num=1, width=1, height=1),
]
# fmt: off
doc.content_boxes = [
    TextBox(doc=doc, page_num=0, x0=0.1, y0=0.1, x1=0.5, y1=0.2, label="body", text="Begin"),  # noqa: E501
    TextBox(doc=doc, page_num=0, x0=0.6, y0=0.1, x1=0.7, y1=0.2, label="body", text="and"),  # noqa: E501
    TextBox(doc=doc, page_num=0, x0=0.8, y0=0.1, x1=0.9, y1=0.2, label="body", text="end."),  # noqa: E501
    TextBox(doc=doc, page_num=1, x0=0.8, y0=0.1, x1=0.9, y1=0.2, label="body", text="New page"),  # noqa: E501
]  # fmt: on

aggregator = SimpleAggregator()
assert aggregator(doc).aggregated_texts == dict(
    body=Text(
        text="Begin and end.\n\nNew page",
    )
)
