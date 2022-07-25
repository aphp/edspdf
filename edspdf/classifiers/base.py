from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, lines: pd.DataFrame) -> List[str]:
        """
        Handles the classification.
        """

    def __call__(self, lines: pd.DataFrame) -> List[str]:
        return self.predict(lines)
