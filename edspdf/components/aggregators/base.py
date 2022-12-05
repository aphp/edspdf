from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, lines: pd.DataFrame) -> Dict[str, str]:
        """
        Handles the text aggregation
        """

    def __call__(self, lines: pd.DataFrame, copy: bool = False) -> Dict[str, str]:
        if copy:
            lines = lines.copy()
        return self.aggregate(lines)
