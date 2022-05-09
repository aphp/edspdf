from abc import ABC, abstractmethod

import pandas as pd


class BaseTransform(ABC):
    @abstractmethod
    def transform(self, lines: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the transformation
        """

    def __call__(self, lines: pd.DataFrame) -> pd.DataFrame:
        return self.transform(lines)
