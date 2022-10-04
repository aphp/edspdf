from abc import ABC, abstractmethod

import pandas as pd


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, pdf: bytes) -> pd.DataFrame:
        """
        Handles the extraction
        """

    def __call__(self, pdf: bytes) -> pd.DataFrame:
        return self.extract(pdf)
