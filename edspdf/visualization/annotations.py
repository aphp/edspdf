from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from PIL.PpmImagePlugin import PpmImageFile

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
    annotations: pd.DataFrame,
    colors: Optional[Union[Dict[str, str], List[str]]] = None,
) -> List[PpmImageFile]:

    pdf_doc = pdfium.PdfDocument(pdf)
    pages = list(pdf_doc.render_topil(scale=2))

    if colors is None:
        colors = {
            key: color
            for key, color in zip(sorted(annotations.label.unique()), CATEGORY20)
        }
    elif isinstance(colors, list):
        colors = {label: color for label, color in zip(colors, CATEGORY20)}

    for page, img in enumerate(pages):
        anns = annotations[annotations.page == page]

        w, h = img.size
        draw = ImageDraw.Draw(img)

        for _, bloc in anns.iterrows():
            draw.rectangle(
                [(bloc.x0 * w, bloc.y0 * h), (bloc.x1 * w, bloc.y1 * h)],
                outline=colors[bloc.label],
                width=3,
            )

    return pages


def compare_results(
    pdf: bytes,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    colors: Optional[Union[Dict[str, str], List[str]]] = None,
) -> List[PpmImageFile]:

    if colors is None:
        colors = list(
            set(list(predictions.label.unique()) + list(labels.label.unique()))
        )

    pages_preds = show_annotations(pdf, predictions, colors)
    pages_labels = show_annotations(pdf, labels, colors)

    pages = []

    for pred, label in zip(pages_preds, pages_labels):
        array = np.hstack((np.asarray(pred), np.asarray(label)))
        pages.append(Image.fromarray(array))

    return pages
