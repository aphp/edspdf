import pandas as pd
from pytest import fixture

from edspdf.aggregation import StyledAggregator
from edspdf.extraction import PdfMinerExtractor


@fixture
def lines(pdf):
    extractor = PdfMinerExtractor()
    df = extractor(pdf)
    df["label"] = "body"
    return df


def test_simple_aggregation(lines):
    aggregator = StyledAggregator()
    text, style = aggregator(lines)

    assert set(text.keys()) == {"body"}
    assert isinstance(style["body"], list)
