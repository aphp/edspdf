from typing import Dict, List, Optional, Union

import numpy as np

from edspdf import PDFDoc, Pipeline, registry


@registry.factory.register("random_classifier")
class RandomClassifier:
    """
    Random classifier, for chaos purposes. Classifies each line to a random element.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        name: str,
        labels: Union[List[str], Dict[str, float]],
        seed: Optional[int] = 0,
    ) -> None:
        super().__init__()

        if isinstance(labels, list):
            labels = {c: 1 for c in labels}

        self.labels = {c: w / sum(labels.values()) for c, w in labels.items()}

        self.rgn = np.random.default_rng(seed=seed)

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        lines = doc.text_boxes
        prediction = self.rgn.choice(
            list(self.labels.keys()),
            p=list(self.labels.values()),
            size=len(lines),
        )
        for b, label in zip(lines, prediction):
            b.label = label

        return doc
