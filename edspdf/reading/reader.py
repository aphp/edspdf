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
        transform: Optional[BaseTransform] = None,
        classifier: Optional[BaseClassifier] = None,
        aggregator: Optional[BaseAggregator] = None,
        meta_labels: Dict[str, str] = dict(),
    ) -> None:
        """
        Reads a text-based PDF document,

        Parameters
        ----------
        extractor : BaseExtractor
            Text bloc extractor.
        transform : Optional[BaseTransform], optional
            Transformation to apply before classification.
        classifier : Optional[BaseClassifier], optional
            Classifier model, to assign a section (eg `body`, `header`, etc).
        aggregator : Optional[BaseAggregator], optional
            Aggregator model, to compile labelled text blocs together.
        meta_labels : Dict[str, str], optional
            Dictionary of hierarchical labels
            (eg `table` is probably within the `body`).
        """

        self.extractor = extractor
        self.classifier = classifier
        self.aggregator = aggregator

        self.transform = transform
        self.meta_labels = meta_labels

    def predict(self, lines: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the label of each text bloc.

        Parameters
        ----------
        lines : pd.DataFrame
            Text blocs to label.

        Returns
        -------
        pd.DataFrame
            Labelled text blocs.
        """

        lines["label"] = self.classifier.predict(lines)
        lines["meta_label"] = lines.label.replace(self.meta_labels)

        return lines

    def prepare_data(self, pdf: bytes, **context: Any) -> pd.DataFrame:
        """
        Prepare data before classification.
        Can also be used to generate the training dataset for the classifier.

        Parameters
        ----------
        pdf : bytes
            PDF document, as bytes.

        Returns
        -------
        pd.DataFrame
            Text blocs as a pandas DataFrame.
        """

        lines = self.extractor(pdf)

        for key, value in context.items():
            lines[key] = value

        # Apply transformation
        if self.transform is not None:
            lines = self.transform(lines)

        return lines

    def prepare_and_predict(self, pdf: bytes, **context: Any) -> pd.DataFrame:
        lines = self.prepare_data(pdf, **context)
        lines = self.predict(lines)
        return lines

    def __call__(self, pdf: bytes, **context: Any) -> Dict[str, str]:
        """
        Process the PDF document.

        Parameters
        ----------
        pdf : bytes
            Byte representation of the PDF document.

        context : Any
            Any contextual information that is used by the classifier
            (eg document type or source).

        Returns
        -------
        Dict[str, str]
            Dictionary containing the aggregated text.
        """
        lines = self.prepare_and_predict(pdf, **context)
        result = self.aggregator(lines)
        return result
