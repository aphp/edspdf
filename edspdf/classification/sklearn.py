from pathlib import Path
from typing import List

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from edspdf.reg import registry

from .base import BaseClassifier


@registry.classifiers.register("sklearn-pipeline.v1")
def sklearn_factory(path: Path) -> "SklearnClassifier":

    pipeline = load(path)

    return SklearnClassifier(pipeline=pipeline)


class SklearnClassifier(BaseClassifier):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict(self, lines: pd.DataFrame) -> List[str]:
        return self.pipeline.predict(lines)
