from itertools import cycle

from edspdf.pipes.aggregators.simple import SimpleAggregator
from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor
from edspdf.structures import Page, PDFDoc, TextBox


def test_no_style():
    doc = PDFDoc(
        content=b"",
        pages=[],
    )
    doc.pages = [
        Page(doc=doc, page_num=0, width=1, height=1),
        Page(doc=doc, page_num=1, width=1, height=1),
    ]
    doc.content_boxes = [
        TextBox(
            doc=doc,
            page_num=0,
            x0=0.1,
            y0=0.1,
            x1=0.5,
            y1=0.2,
            label="body",
            text="Begin",
        ),
        TextBox(
            doc=doc,
            page_num=0,
            x0=0.6,
            y0=0.1,
            x1=0.7,
            y1=0.2,
            label="body",
            text="and",
        ),
        TextBox(
            doc=doc,
            page_num=0,
            x0=0.8,
            y0=0.1,
            x1=0.9,
            y1=0.2,
            label="body",
            text="end.",
        ),
        TextBox(
            doc=doc,
            page_num=1,
            x0=0.8,
            y0=0.1,
            x1=0.9,
            y1=0.2,
            label="body",
            text="New page",
        ),
    ]

    aggregator = SimpleAggregator()
    assert aggregator(doc).aggregated_texts["body"].text == "Begin and end.\n\nNew page"


def test_styled_pdfminer_aggregation(styles_pdf):
    extractor = PdfMinerExtractor(extract_style=True)
    aggregator = SimpleAggregator(
        sort=True,
        label_map={
            "header": ["header"],
            "body": "body",
        },
    )

    doc = extractor(styles_pdf)
    for b, label in zip(doc.text_boxes, cycle(["header", "body"])):
        b.label = label
    doc = aggregator(doc)
    texts = {k: v.text for k, v in doc.aggregated_texts.items()}
    props = {k: v.properties for k, v in doc.aggregated_texts.items()}

    assert set(texts.keys()) == {"body", "header"}
    assert isinstance(props["body"], list)

    for value in props.values():
        assert value[0].begin == 0

    pairs = set()
    for label in texts.keys():
        for prop in props[label]:
            pairs.add(
                (
                    texts[label][prop.begin : prop.end],
                    " ".join(
                        filter(
                            bool,
                            (
                                ("italic" if prop.italic else ""),
                                ("bold" if prop.bold else ""),
                            ),
                        )
                    ),
                )
            )

    assert pairs == {
        ("This is a", ""),
        ("test", "bold"),
        ("to check EDS-PDF’s", ""),
        ("ability", "italic"),
        ("to detect changing styles.", ""),
        ("Let’s up the stakes, with", ""),
        ("intra", "italic"),
        ("-word change. Or better yet,", ""),
        ("this mi", "bold"),
        ("ght be hard.", ""),
    }


def test_styled_pdfminer_aggregation_letter(letter_pdf):
    extractor = PdfMinerExtractor(extract_style=True)
    aggregator = SimpleAggregator()

    doc = extractor(letter_pdf)
    for b, label in zip(doc.content_boxes, cycle(["header", "body"])):
        b.label = label
    doc = aggregator(doc)
    texts = {k: v.text for k, v in doc.aggregated_texts.items()}
    props = {k: v.properties for k, v in doc.aggregated_texts.items()}

    assert set(texts.keys()) == {"body", "header"}
    assert isinstance(props["body"], list)

    for value in props.values():
        assert value[0].begin == 0

    pairs = set()
    for label in texts.keys():
        for prop in props[label]:
            pairs.add(
                (
                    texts[label][prop.begin : prop.end],
                    " ".join(
                        filter(
                            bool,
                            (
                                ("italic" if prop.italic else ""),
                                ("bold" if prop.bold else ""),
                            ),
                        )
                    ),
                )
            )
