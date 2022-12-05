from edspdf.components import SimpleAggregator
from edspdf.models import PDFDoc, TextBox

example = PDFDoc(
    content=b"",
    lines=[
        TextBox(page=0, x0=0.1, y0=0.1, x1=0.5, y1=0.2, label="body", text="Begin"),
        TextBox(page=0, x0=0.6, y0=0.1, x1=0.7, y1=0.2, label="body", text="and"),
        TextBox(page=0, x0=0.8, y0=0.1, x1=0.9, y1=0.2, label="body", text="end."),
        TextBox(page=1, x0=0.8, y0=0.1, x1=0.9, y1=0.2, label="body", text="New page"),
    ],
)


def test_simple_aggregation():
    aggregator = SimpleAggregator()
    assert aggregator(example) == dict(body="Begin and end.\n\nNew page")
