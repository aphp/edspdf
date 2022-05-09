from pathlib import Path
from typing import List

import pandas as pd
from joblib import load

from edspdf.reg import registry

from .base import BaseClassifier


@registry.classifiers.register("sklearn-pipeline.v1")
class SklearnClassifier(BaseClassifier):
    def __init__(self, path: Path):
        self.pipeline = load(path)

    def predict(self, lines: pd.DataFrame) -> List[str]:
        return self.pipeline.predict(lines)
