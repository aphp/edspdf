from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from PIL.PpmImagePlugin import PpmImageFile

from edspdf.models import Box

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

    pdf_doc = pdfium.PdfDocument(pdf)
    pages = list(pdf_doc.render_topil(scale=2))
    unique_labels = list(dict.fromkeys([box.label for box in annotations]))

    if colors is None:
        colors = {key: color for key, color in zip(unique_labels, CATEGORY20)}
    elif isinstance(colors, list):
        colors = {label: color for label, color in zip(colors, CATEGORY20)}

    for page, img in enumerate(pages):

        w, h = img.size
        draw = ImageDraw.Draw(img)

        for bloc in annotations:
            if bloc.page == page:
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

    if colors is None:
        colors = list(
            {
                **dict.fromkeys([b.label for b in pred]),
                **dict.fromkeys([b.label for b in gold]),
            }
        )

    pages_pred = show_annotations(pdf, pred, colors)
    pages_gold = show_annotations(pdf, gold, colors)

    pages = []

    for page_pred, page_gold in zip(pages_pred, pages_gold):
        array = np.hstack((np.asarray(page_pred), np.asarray(page_gold)))
        pages.append(Image.fromarray(array))

    return pages
