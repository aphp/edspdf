from typing import Sequence

from edspdf import Box, PDFDoc, Pipeline, registry
from edspdf.utils.alignment import align_box_labels


@registry.factory.register("mask-classifier")
def simple_mask_classifier_factory(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    pipeline: Pipeline = None,
    name: str = "mask-classifier",
    threshold: float = 1.0,
):
    """
    The simplest form of mask classification. You define the mask, everything else
    is tagged as pollution.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline object
    name: str
        The name of the component
    x0: float
        The x0 coordinate of the mask
    y0: float
        The y0 coordinate of the mask
    x1: float
        The x1 coordinate of the mask
    y1: float
        The y1 coordinate of the mask
    threshold: float
        The threshold for the alignment

    Examples
    --------

    === "API-based"

        ```python
        pipeline.add_pipe(
            "mask-classifier",
            name="classifier",
            config={
                "threshold": 0.9,
                "x0": 0.1,
                "y0": 0.1,
                "x1": 0.9,
                "y1": 0.9,
            },
        )
        ```

    === "Configuration-based"

        ```toml
        [components.classifier]
        @classifiers = "mask-classifier"
        x0 = 0.1
        y0 = 0.1
        x1 = 0.9
        y1 = 0.9
        threshold = 0.9
        ```
    """
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


@registry.factory.register("multi-mask-classifier")
def mask_classifier_factory(
    pipeline: Pipeline = None,
    name: str = "multi-mask-classifier",
    threshold: float = 1.0,
    **masks: Box,
):
    """
    A generalisation, wherein the user defines a number of regions.

    The following configuration produces _exactly_ the same classifier as `mask.v1`
    example above.

    Any bloc that is not part of a mask is tagged as `pollution`.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline object
    name: str
    threshold: float
        The threshold for the alignment
    masks: Dict[str, Box]
        The masks

    Examples
    --------

    === "API-based"

        ```python
        pipeline.add_pipe(
            "multi-mask-classifier",
            name="classifier",
            config={
                "threshold": 0.9,
                "mymask": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.3, "label": "body"},
            },
        )
        ```

    === "Configuration-based"

        ```toml
        [components.classifier]
        @factory = "multi-mask-classifier"
        threshold = 0.9

        [components.classifier.mymask]
        label = "body"
        x0 = 0.1
        y0 = 0.1
        x1 = 0.9
        y1 = 0.9
        ```

    The following configuration defines a `header` region.

    === "API-based"

        ```python
        pipeline.add_pipe(
            "multi-mask-classifier",
            name="classifier",
            config={
                "threshold": 0.9,
                "body": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.3, "label": "header"},
                "header": {"x0": 0.1, "y0": 0.3, "x1": 0.9, "y1": 0.9, "label": "body"},
            },
        )
        ```

    === "Configuration-based"

        ```toml
        [components.classifier]
        @factory = "multi-mask-classifier"
        threshold = 0.9

        [components.classifier.header]
        label = "header"
        x0 = 0.1
        y0 = 0.1
        x1 = 0.9
        y1 = 0.3

        [components.classifier.body]
        label = "body"
        x0 = 0.1
        y0 = 0.3
        x1 = 0.9
        y1 = 0.9
        ```
    """
    return MaskClassifier(
        pipeline=pipeline,
        name=name,
        masks=list(masks.values()),
        threshold=threshold,
    )


class MaskClassifier:
    """
    Simple mask classifier, that labels every box inside one of the masks
    with its label.
    """

    def __init__(
        self,
        pipeline: Pipeline = None,
        name: str = "multi-mask-classifier",
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
