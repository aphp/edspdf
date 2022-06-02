import re

import pandas as pd

from edspdf.reg import registry

from .base import BaseTransform


@registry.transforms.register("chain.v1")
class ChainTransform(BaseTransform):
    def __init__(self, *layers: BaseTransform):
        self.layers = layers

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for layer in self.layers:
            df = layer(df)
        return df


@registry.transforms.register("telephone.v1")
class AddPhone(BaseTransform):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["telephone"] = df.text.apply(
            lambda t: len(list(re.finditer(r"\b(\d\d[\s\.]?){5}\b", t)))
        )
        return df


@registry.transforms.register("dates.v1")
class AddDates(BaseTransform):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df["date"] = df.text.apply(
            lambda t: len(list(re.finditer(r"\b\d\d\/\d\d\/?(\d{4})?\b", t)))
        )

        return df


@registry.transforms.register("dimensions.v1")
class AddDimensions(BaseTransform):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df["width"] = df.x1 - df.x0
        df["height"] = df.y1 - df.y0
        df["area"] = df.width * df.height

        return df


@registry.transforms.register("rescale.v1")
class Rescale(BaseTransform):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df["x0_"] = df.x0 * df.page_width
        df["x1_"] = df.x1 * df.page_width

        df["y0_"] = df.y0 * df.page_height
        df["y1_"] = df.y1 * df.page_height

        return df
