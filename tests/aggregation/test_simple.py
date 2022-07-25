import pandas as pd

from edspdf.aggregators.simple import SimpleAggregator

example = pd.DataFrame.from_records(
    [
        dict(page=0, x0=0.1, y0=0.1, x1=0.5, y1=0.2, label="body", text="Begin"),
        dict(page=0, x0=0.6, y0=0.1, x1=0.7, y1=0.2, label="body", text="and"),
        dict(page=0, x0=0.8, y0=0.1, x1=0.9, y1=0.2, label="body", text="end."),
    ]
)


def test_simple_aggregation():
    aggregator = SimpleAggregator()
    assert aggregator(example, copy=True) == dict(body="Begin and end.\n\n")
