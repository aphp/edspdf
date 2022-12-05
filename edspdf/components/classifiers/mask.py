from edspdf import Component, registry
from edspdf.models import Box, PDFDoc
from edspdf.utils.alignment import align_box_labels


@registry.factory.register("mask-classifier")
def simple_mask_classifier_factory(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    threshold: float = 1.0,
):
    return MaskClassifier(
        Box(
            label="body",
            page=None,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        ),
        threshold=threshold,
    )


@registry.factory.register("multi-mask-classifier")
def mask_classifier_factory(threshold: float = 1.0, **masks: Box):
    return MaskClassifier(*masks.values(), threshold=threshold)


class MaskClassifier(Component):
    """
    Mask classifier, that reproduces the PdfBox behaviour.
    """

    def __init__(
        self,
        *ms: Box,
        threshold: float = 1.0,
    ):
        super().__init__()

        masks = list(ms)

        masks.append(
            Box(
                label="pollution",
                x0=-10000,
                x1=10000,
                y0=-10000,
                y1=10000,
            )
        )

        self.masks = masks
        self.threshold = threshold

    def __call__(self, doc: PDFDoc) -> PDFDoc:

        doc.lines = align_box_labels(
            src_boxes=self.masks,
            dst_boxes=doc.lines,
            threshold=self.threshold,
        )

        return doc
