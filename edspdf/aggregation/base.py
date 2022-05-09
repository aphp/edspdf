from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, lines) -> Dict[str, str]:
        pass

    def __call__(self, lines: pd.DataFrame) -> Dict[str, str]:
        return self.aggregate(lines)
