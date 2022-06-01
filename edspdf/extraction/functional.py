import re
from typing import Iterator, Tuple

import pandas as pd
from pdfminer.high_level import LTPage
from pdfminer.layout import LTTextBoxHorizontal

from .models import Line
from .style import extract_style

MULTISPACE_PATTERN = re.compile(r"\s+")


def get_blocs(
    layout: Iterator[LTPage],
) -> Iterator[Tuple[LTTextBoxHorizontal, int, float, float]]:
    """
    Extract text blocs from a PDFMiner layout generator.

    Arguments
    ---------
    layout:
        PDFMiner layout generator.

    Yields
    ------
    bloc :
        Text bloc
    """

    for i, page in enumerate(layout):

        width = page.width
        height = page.height

        for bloc in page:
            if isinstance(bloc, LTTextBoxHorizontal):
                yield bloc, i, width, height


# def extract_text(line: LTTextBoxHorizontal) -> str:

#     text = line.get_text()
#     text = MULTISPACE_PATTERN.sub(" ", text)
#     text = text.strip()

#     return text


def get_lines(layout: Iterator[LTPage]) -> Iterator[Line]:
    """
    Extract lines from a PDFMiner layout object.

    The line is reframed such that the origin is the top left corner.

    Parameters
    ----------
    layout : Iterator[LTPage]
        PDFMiner layout object.

    Yields
    -------
    Iterator[Line]
        Single line object.
    """
    for b, (bloc, p, w, h) in enumerate(get_blocs(layout)):
        for line in bloc:
            text, styles = extract_style(line, width=w, height=h)
            yield Line(
                page=p,
                bloc=b,
                x0=line.x0 / w,
                x1=line.x1 / w,
                y0=1 - line.y1 / h,
                y1=1 - line.y0 / h,
                page_width=w,
                page_height=h,
                text=text,
                styles=styles,
            )


# def extract_styled_text(lines: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add a `styled_text` column to a dataframe of extracted lines.

#     Parameters
#     ----------
#     lines : pd.DataFrame
#         Dataframe containing extracted lines.

#     Returns
#     -------
#     pd.DataFrame
#         Dataframe with the `styled_text` column added.
#     """

#     styled_text = lines.line.apply(line2style)
#     lines["styled_text"] = styled_text

#     return lines


def remove_outside_lines(
    lines: pd.DataFrame,
    strict_mode: bool = False,
) -> pd.DataFrame:
    """
    Filter out lines that are outside the canvas.

    Parameters
    ----------
    lines : pd.DataFrame
        Dataframe of extracted lines
    strict_mode : bool, optional
        Whether to remove the line if any part of it is outside the canvas,
        by default False

    Returns
    -------
    pd.DataFrame
        Filtered lines.
    """
    if strict_mode:
        lower = lines[["x0", "y0"]].min(axis=1) >= 0
        upper = lines[["x1", "y1"]].max(axis=1) <= 1
        lines = lines[lower & upper]
    else:
        below = lines[["x1", "y1"]].max(axis=1) < 0
        above = lines[["x0", "y0"]].min(axis=1) > 0
        lines = lines[~(below | above)]
    return lines
