from typing import List

import pandas as pd

from edspdf.utils.registry import registry


@registry.classifiers.register("dummy.v1")
class DummyClassifier(object):
    """
    "Dummy" classifier, for testing purposes. Classifies every line to ``body``.
    """

    def predict(self, lines: pd.DataFrame) -> List[str]:
        return ["body"] * len(lines)
