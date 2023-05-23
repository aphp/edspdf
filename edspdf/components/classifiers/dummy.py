from edspdf.pipeline import Pipeline
from edspdf.registry import registry
from edspdf.structures import PDFDoc


@registry.factory.register("dummy_classifier")
class DummyClassifier:
    """
    Dummy classifier, for chaos purposes. Classifies each line to a random element.
    """

    def __init__(
        self,
        label: str,
        pipeline: Pipeline = None,
        name: str = "dummy_classifier",
    ) -> None:
        self.name = name
        self.label = label

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        for b in doc.lines:
            b.label = self.label

        return doc
