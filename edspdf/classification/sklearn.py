from importlib import resources
from pathlib import Path
from typing import List, Optional

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from edspdf.reg import registry

from .base import BaseClassifier


@registry.classifiers.register("sklearn-pipeline.v1")
def sklearn_factory(path: Path, package: Optional[Path] = None) -> "SklearnClassifier":

    if package is not None:
        with resources.path(package=package, resource="__init__.py") as p:
            path = p.parent / path

    pipeline = load(path)

    return SklearnClassifier(pipeline=pipeline)


class SklearnClassifier(BaseClassifier):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict(self, lines: pd.DataFrame) -> List[str]:
        return self.pipeline.predict(lines)
