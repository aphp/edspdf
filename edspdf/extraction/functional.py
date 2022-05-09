from typing import Generator, Iterator, Tuple

import pandas as pd
from pdfminer.high_level import LTPage
from pdfminer.layout import LTTextBoxHorizontal

from edspdf.extraction.style import line2style

from .models import Line


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


def get_lines(layout: Iterator[LTPage]) -> Generator[Line, None, None]:
    """
    Extract lines from a PDFMiner layout object.

    The line is reframed such that the origin is the top left corner.

    Parameters
    ----------
    layout : Iterator[LTPage]
        PDFMiner layout object.

    Yields
    -------
    Generator[Line, None, None]
        Single line object.
    """
    for b, (bloc, p, w, h) in enumerate(get_blocs(layout)):
        for line in bloc:
            yield Line(
                page=p,
                bloc=b,
                line=line,
                x0=line.x0 / w,
                x1=line.x1 / w,
                y0=1 - line.y1 / h,
                y1=1 - line.y0 / h,
                page_width=w,
                page_height=h,
            )


def extract_text(lines: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``text`` column to a dataframe containing lines,
    using PDFMiner's ``get_text`` method.

    Parameters
    ----------
    lines : pd.DataFrame
        Dataframe containing extracted lines.

    Returns
    -------
    pd.DataFrame
        Dataframe with the ``text`` column added.
    """

    text = lines.line.apply(lambda line: line.get_text())
    text = text.str.replace(r"\s+", " ", regex=True)
    text = text.str.strip()

    lines["text"] = text

    return lines


def extract_styled_text(lines: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``styled_text`` column to a dataframe of extracted lines.

    Parameters
    ----------
    lines : pd.DataFrame
        Dataframe containing extracted lines.

    Returns
    -------
    pd.DataFrame
        Dataframe with the ``styled_text`` column added.
    """

    styled_text = lines.line.apply(line2style)
    lines["styled_text"] = styled_text

    return lines


def remove_outside_lines(
    lines: pd.DataFrame,
    strict_mode: bool = False,
    copy: bool = False,
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
    copy : bool, optional
        Whether to copy the dataframe before filtering it,
        by default False

    Returns
    -------
    pd.DataFrame
        Filtered lines.
    """
    if strict_mode:
        lines = lines[lines[["x0", "y0", "x1", "y1"]].min(axis=1) > 0]
    else:
        lines = lines[
            (lines[["x0", "x1"]].max(axis=1) > 0)
            & (lines[["y0", "y1"]].max(axis=1) > 0)
        ]
    return lines
