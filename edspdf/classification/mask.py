from typing import List

import pandas as pd
from pydantic import BaseModel, Field, confloat, parse_obj_as, validator

from edspdf.reg import registry

from .align import align_labels
from .base import BaseClassifier


class Mask(BaseModel):

    label: str

    X0: confloat(ge=0, le=1) = Field(0, alias="x0")
    X1: confloat(ge=0, le=1) = Field(1, alias="x1")
    Y0: confloat(ge=0, le=1) = Field(0, alias="y0")
    Y1: confloat(ge=0, le=1) = Field(1, alias="y1")

    threshold: confloat(ge=0, le=1) = 1

    @validator("X1", always=True)
    def check_x(cls, v, values):
        if v <= values["X0"]:
            raise ValueError("x1 should be greater than x0")
        return v

    @validator("Y1", always=True)
    def check_y(cls, v, values):
        if v <= values["Y0"]:
            raise ValueError("y1 should be greater than y0")
        return v


@registry.classifiers.register("mask.v1")
def simple_mask_classifier_factory(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    threshold: float = 1.0,
):
    return MaskClassifier(
        Mask(
            label="body",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            threshold=threshold,
        )
    )


@registry.classifiers.register("custom_masks.v1")
def mask_classifier_factory(**masks):
    return MaskClassifier(*parse_obj_as(List[Mask], list(masks.values())))


class MaskClassifier(BaseClassifier):
    """
    Mask classifier, that reproduces the PdfBox behaviour.
    """

    def __init__(
        self,
        *ms: Mask,
    ) -> None:

        masks = list(ms)

        masks.append(Mask(label="pollution"))

        self.comparison = pd.DataFrame.from_records([mask.dict() for mask in masks])

    def predict(self, lines: pd.DataFrame) -> pd.Series:

        df = align_labels(lines, self.comparison)

        return df.label
