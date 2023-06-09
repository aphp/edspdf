from edspdf.pipeline import Pipeline
from edspdf.registry import registry
from edspdf.structures import PDFDoc


@registry.factory.register("dummy-classifier")
class DummyClassifier:
    """
    Dummy classifier, for chaos purposes. Classifies each line to a random element.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline object.
    name: str
        The name of the component.
    label: str
        The label to assign to each line.
    """

    def __init__(
        self,
        label: str,
        pipeline: Pipeline = None,
        name: str = "dummy-classifier",
    ) -> None:
        self.name = name
        self.label = label

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        for b in doc.content_boxes:
            b.label = self.label

        return doc
