import pandas as pd
from pytest import fixture


@fixture
def lines() -> pd.DataFrame:
    df = pd.DataFrame.from_records(
        [
            dict(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
            dict(x0=0.1, y0=0.6, x1=0.4, y1=0.7),
            dict(x0=0.1, y0=0.6, x1=0.9, y1=0.7),
        ]
    )

    df["page_width"] = 1
    df["page_height"] = 1

    return df
