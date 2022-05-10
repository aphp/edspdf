from typing import List

import pandas as pd
import pdf2image
from PIL import ImageDraw
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


def show_annotations(pdf: bytes, annotations: pd.DataFrame) -> List[PpmImageFile]:
    pages = pdf2image.convert_from_bytes(pdf)
    colors = {key: color for key, color in zip(annotations.label.unique(), CATEGORY20)}

    for page, img in enumerate(pages):
        anns = annotations[annotations.page == page]

        w, h = img.size
        draw = ImageDraw.Draw(img)

        for _, bloc in anns.iterrows():
            draw.rectangle(
                [(bloc.x0 * w, bloc.y0 * h), (bloc.x1 * w, bloc.y1 * h)],
                outline=colors[bloc.label],
                width=4,
            )

    return pages
