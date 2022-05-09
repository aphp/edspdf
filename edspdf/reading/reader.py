from typing import Any, Dict, Optional

import pandas as pd

from edspdf.aggregation import BaseAggregator
from edspdf.classification import BaseClassifier
from edspdf.extraction import BaseExtractor
from edspdf.reg import registry
from edspdf.transforms import BaseTransform


@registry.readers.register("pdf-reader.v1")
class PdfReader:
    def __init__(
        self,
        extractor: BaseExtractor,
        classifier: BaseClassifier,
        aggregator: BaseAggregator,
        meta_labels: Dict[str, str] = dict(),
        transform: Optional[BaseTransform] = None,
    ) -> None:

        self.extractor = extractor
        self.classifier = classifier
        self.aggregator = aggregator

        self.transform = transform
        self.meta_labels = meta_labels

    def predict(self, lines: pd.DataFrame, copy: bool = True) -> pd.DataFrame:

        if copy:
            lines = lines.copy()

        lines["label"] = self.classifier.predict(lines)
        lines["meta_label"] = lines.label.replace(self.meta_labels)

        return lines

    def prepare_data(self, pdf: bytes, **context: Any) -> Optional[str]:

        lines = self.extractor(pdf)

        for key, value in context.items():
            lines[key] = value

        # Apply transformation
        if self.transform is not None:
            lines = self.transform(lines)

        return lines

    def __call__(self, pdf: bytes, **context: Any) -> Dict[str, str]:
        lines = self.prepare_data(pdf, **context)
        lines = self.predict(lines)
        result = self.aggregator(lines)
        return result
