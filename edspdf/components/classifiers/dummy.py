from edspdf import Component, registry
from edspdf.models import PDFDoc


@registry.factory.register("dummy-classifier")
class DummyClassifier(Component):
    """
    Dummy classifier, for chaos purposes. Classifies each line to a random element.
    """

    def __init__(
        self,
        label: str,
    ) -> None:
        super().__init__()

        self.label = label

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        for b in doc.lines:
            b.label = self.label

        return doc
