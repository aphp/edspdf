from itertools import cycle

from edspdf.components.aggregators.styled import StyledAggregator
from edspdf.components.extractors.pdfminer import PdfMinerExtractor


def test_styled_pdfminer_aggregation(styles_pdf):
    extractor = PdfMinerExtractor(extract_properties=True)
    aggregator = StyledAggregator()

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
    extractor = PdfMinerExtractor(extract_properties=True)
    aggregator = StyledAggregator()

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
