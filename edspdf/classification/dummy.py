from typing import List

import pandas as pd

from edspdf.reg import registry

from .base import BaseClassifier


@registry.classifiers.register("dummy.v1")
class DummyClassifier(BaseClassifier):
    """
    "Dummy" classifier, for testing purposes. Classifies every line to ``body``.
    """

    def predict(self, lines: pd.DataFrame) -> List[str]:
        return ["body"] * len(lines)
