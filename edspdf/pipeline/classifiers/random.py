from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from edspdf.reg import registry

from .base import BaseClassifier


@registry.classifiers.register("random.v1")
class RandomClassifier(BaseClassifier):
    """
    Random classifier, for chaos purposes. Classifies each line to a random element.
    """

    def __init__(
        self,
        classes: Union[List[str], Dict[str, float]],
        seed: Optional[int] = 0,
    ) -> None:

        if isinstance(classes, list):
            classes = {c: 1 for c in classes}

        self.classes = {c: w / sum(classes.values()) for c, w in classes.items()}

        self.rgn = np.random.default_rng(seed=seed)

    def predict(self, lines: pd.DataFrame) -> List[str]:
        choices = self.rgn.choice(
            list(self.classes.keys()),
            p=list(self.classes.values()),
            size=len(lines),
        )

        return list(choices)
