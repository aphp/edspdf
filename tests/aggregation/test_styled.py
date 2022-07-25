from itertools import cycle

import pandas as pd
from pytest import fixture

from edspdf.aggregators import StyledAggregator
from edspdf.extractors import PdfMinerExtractor


@fixture
def lines(pdf):
    extractor = PdfMinerExtractor()
    df = extractor(pdf)
    df["label"] = [
        label for _, label in zip(range(len(df)), cycle(["header", "footer", "body"]))
    ]
    return df


def test_simple_aggregation(lines):
    aggregator = StyledAggregator()
    text, style = aggregator(lines)

    assert set(text.keys()) == {"body", "header", "footer"}
    assert isinstance(style["body"], list)

    for value in style.values():
        assert value[0]["start"] == 0
