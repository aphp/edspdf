from itertools import cycle

from edspdf.components import PdfMinerExtractor, StyledAggregator


def test_styled_pdfminer_aggregation(styles_pdf):
    extractor = PdfMinerExtractor(extract_style=True)
    aggregator = StyledAggregator()

    doc = extractor(styles_pdf)
    for b, label in zip(doc.lines, cycle(["header", "body"])):
        b.label = label
    texts, styles = aggregator(doc)

    assert set(texts.keys()) == {"body", "header"}
    assert isinstance(styles["body"], list)

    for value in styles.values():
        assert value[0]["begin"] == 0

    pairs = set()
    for label in texts.keys():
        for style in styles[label]:
            pairs.add(
                (
                    texts[label][style["begin"] : style["end"]],
                    " ".join(
                        filter(
                            bool,
                            (
                                ("italic" if style["italic"] else ""),
                                ("bold" if style["bold"] else ""),
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
    aggregator = StyledAggregator()

    doc = extractor(letter_pdf)
    for b, label in zip(doc.lines, cycle(["header", "body"])):
        b.label = label
    texts, styles = aggregator(doc)

    assert set(texts.keys()) == {"body", "header"}
    assert isinstance(styles["body"], list)

    for value in styles.values():
        assert value[0]["begin"] == 0

    pairs = set()
    for label in texts.keys():
        for style in styles[label]:
            pairs.add(
                (
                    texts[label][style["begin"] : style["end"]],
                    " ".join(
                        filter(
                            bool,
                            (
                                ("italic" if style["italic"] else ""),
                                ("bold" if style["bold"] else ""),
                            ),
                        )
                    ),
                )
            )
