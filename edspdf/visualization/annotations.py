from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from PIL.PpmImagePlugin import PpmImageFile

from edspdf.structures import Box

CATEGORY20 = [
    "#1f77b4",
    # "#aec7e8",
    "#ff7f0e",
    # "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]


def show_annotations(
    pdf: bytes,
    annotations: Sequence[Box],
    colors: Optional[Union[Dict[str, str], List[str]]] = None,
) -> List[PpmImageFile]:
    """
    Show Box annotations on a PDF document.

    Parameters
    ----------
    pdf: bytes
        Bytes content of the PDF document
    annotations: Sequence[Box]
        List of Box annotations to show
    colors: Optional[Union[Dict[str, str], List[str]]]
        Colors to use for each label. If a list is provided, it will be used to color
        the first `len(colors)` unique labels. If a dictionary is provided, it will be
        used to color the labels in the dictionary. If None, a default color scheme will
        be used.

    Returns
    -------
    List[PpmImageFile]
        List of PIL images with the annotations. You can display them in a notebook
        with `display(*pages)`.
    """

    pdf_doc = pdfium.PdfDocument(pdf)
    pages = list([page.render(scale=2).to_pil() for page in pdf_doc])
    unique_labels = list(dict.fromkeys([box.label for box in annotations]))

    if colors is None:
        colors = {key: color for key, color in zip(unique_labels, CATEGORY20)}
    elif isinstance(colors, list):
        colors = {label: color for label, color in zip(unique_labels, colors)}

    for page_num, img in enumerate(pages):

        w, h = img.size
        draw = ImageDraw.Draw(img)

        for bloc in annotations:
            if bloc.page_num == page_num:
                draw.rectangle(
                    [(bloc.x0 * w, bloc.y0 * h), (bloc.x1 * w, bloc.y1 * h)],
                    outline=colors[bloc.label],
                    width=3,
                )

    return pages


def compare_results(
    pdf: bytes,
    pred: Sequence[Box],
    gold: Sequence[Box],
    colors: Optional[Union[Dict[str, str], List[str]]] = None,
) -> List[PpmImageFile]:
    """
    Compare two sets of annotations on a PDF document.

    Parameters
    ----------
    pdf: bytes
        Bytes content of the PDF document
    pred: Sequence[Box]
        List of Box annotations to show on the left side
    gold: Sequence[Box]
        List of Box annotations to show on the right side
    colors: Optional[Union[Dict[str, str], List[str]]]
        Colors to use for each label. If a list is provided, it will be used to color
        the first `len(colors)` unique labels. If a dictionary is provided, it will be
        used to color the labels in the dictionary. If None, a default color scheme will
        be used.

    Returns
    -------
    List[PpmImageFile]
        List of PIL images with the annotations. You can display them in a notebook
        with `display(*pages)`.
    """
    if colors is None:
        colors = {
            **dict.fromkeys([b.label for b in pred]),
            **dict.fromkeys([b.label for b in gold]),
        }

    pages_pred = show_annotations(pdf, pred, colors)
    pages_gold = show_annotations(pdf, gold, colors)

    pages = []

    for page_pred, page_gold in zip(pages_pred, pages_gold):
        array = np.hstack((np.asarray(page_pred), np.asarray(page_gold)))
        pages.append(Image.fromarray(array))

    return pages
