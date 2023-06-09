from typing import Dict, List, Optional, Union

import numpy as np

from edspdf import PDFDoc, Pipeline, registry


@registry.factory.register("random-classifier")
class RandomClassifier:
    """
    Random classifier, for chaos purposes. Classifies each box to a random element.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline object.
    name: str
        The name of the component.
    labels: Union[List[str], Dict[str, float]]
        The labels to assign to each line. If a list is passed, each label is assigned
        with equal probability. If a dict is passed, the keys are the labels and the
        values are the probabilities.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        labels: Union[List[str], Dict[str, float]],
        seed: Optional[int] = 0,
        name: str = "random-classifier",
    ) -> None:
        super().__init__()

        if isinstance(labels, list):
            labels = {c: 1 for c in labels}

        self.labels = {c: w / sum(labels.values()) for c, w in labels.items()}

        self.rgn = np.random.default_rng(seed=seed)

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        lines = doc.content_boxes
        prediction = self.rgn.choice(
            list(self.labels.keys()),
            p=list(self.labels.values()),
            size=len(lines),
        )
        for b, label in zip(lines, prediction):
            b.label = label

        return doc
