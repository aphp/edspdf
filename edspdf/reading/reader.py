from typing import Any, Optional

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
        transform: Optional[BaseTransform] = None,
    ) -> None:

        self.extractor = extractor
        self.classifier = classifier
        self.aggregator = aggregator

        self.transform = transform

    def predict(self, lines: pd.DataFrame, copy: bool = True) -> pd.DataFrame:

        if copy:
            lines = lines.copy()

        lines["label"] = self.classifier.predict(lines)
        lines["meta_label"] = lines.label.replace(
            {"section_title": "body", "table": "body"}
        )

        return lines

    def process(self, pdf: bytes, **context: Any) -> Optional[str]:

        lines = self.extractor(pdf)

        for key, value in context.items():
            lines[key] = value

        # Apply transformation
        if self.transform is not None:
            lines = self.transform(lines)

        lines = self.predict(lines)

        result = self.aggregator(lines)

        return result

    def __call__(self, pdf: bytes, **context: Any) -> str:
        return self.process(pdf, **context)
