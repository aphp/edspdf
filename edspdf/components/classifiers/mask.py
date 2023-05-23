from typing import Sequence

from edspdf import Box, PDFDoc, Pipeline, registry
from edspdf.utils.alignment import align_box_labels


@registry.factory.register("mask_classifier")
def simple_mask_classifier_factory(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    pipeline: Pipeline = None,
    name: str = "mask_classifier",
    threshold: float = 1.0,
):
    return MaskClassifier(
        pipeline=pipeline,
        name=name,
        masks=[
            Box(
                label="body",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
        ],
        threshold=threshold,
    )


@registry.factory.register("multi_mask_classifier")
def mask_classifier_factory(
    pipeline: Pipeline = None,
    name: str = "mask_classifier",
    threshold: float = 1.0,
    **masks: Box,
):
    return MaskClassifier(
        pipeline=pipeline,
        name=name,
        masks=list(masks.values()),
        threshold=threshold,
    )


class MaskClassifier:
    """
    Mask classifier, that reproduces the PdfBox behaviour.
    """

    def __init__(
        self,
        pipeline: Pipeline = None,
        name: str = "multi_mask_classifier",
        masks: Sequence[Box] = (),
        threshold: float = 1.0,
    ):
        self.name = name

        masks = list(masks)

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
        doc.content_boxes = align_box_labels(
            src_boxes=self.masks,
            dst_boxes=doc.content_boxes,
            threshold=self.threshold,
        )

        return doc
