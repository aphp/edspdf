import re

import pandas as pd


class ChainTransform:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for layer in self.layers:
            df = layer(df)
        return df


def add_telephone(df: pd.DataFrame) -> pd.DataFrame:
    df["telephone"] = df.text.apply(
        lambda t: len(list(re.finditer(r"\b(\d\d[\s\.]?){5}\b", t)))
    )
    return df


def add_dates(df: pd.DataFrame) -> pd.DataFrame:

    df["date"] = df.text.apply(
        lambda t: len(list(re.finditer(r"\b\d\d\/\d\d\/?(\d{4})?\b", t)))
    )

    return df


def add_dimensions(df: pd.DataFrame) -> pd.DataFrame:

    df["width"] = df.x1 - df.x0
    df["height"] = df.y1 - df.y0
    df["area"] = df.width * df.height

    return df
