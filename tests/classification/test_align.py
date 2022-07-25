import pandas as pd

from edspdf.classifiers.align import align_labels


def test_align_multipage(multipage_lines):

    df = pd.DataFrame.from_records(
        [
            dict(X0=0.0, Y0=0.0, X1=1.0, Y1=1.0, page=0, label="big"),
            dict(X0=0.1, Y0=0.1, X1=0.9, Y1=0.9, page=1, label="small"),
        ]
    )

    labelled = align_labels(multipage_lines, df)
    assert list(labelled["label"]) == ["big", "big", "big", "small", "small", "small"]


def test_align_crosspage(multipage_lines):

    df = pd.DataFrame.from_records(
        [
            dict(X0=0.0, Y0=0.0, X1=1.0, Y1=1.0, label="big"),
            dict(X0=0.1, Y0=0.1, X1=0.9, Y1=0.9, label="small"),
        ]
    )

    labelled = align_labels(multipage_lines, df)
    assert list(labelled["label"]) == [
        "small",
        "small",
        "small",
        "small",
        "small",
        "small",
    ]
