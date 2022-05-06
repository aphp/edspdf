from io import BytesIO
from typing import Optional

import pandas as pd
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams

from .functional import (
    extract_styled_text,
    extract_text,
    get_lines,
    remove_outside_lines,
)


class LineExtractor(object):
    """
    Extractor object. Given a PDF byte stream, produces a list of boxes.

    Parameters
    ----------
    laparams : LAParams, optional
        PDFMiner LAParams object.
    style : bool, default: False
        Whether to extract style.
    """

    def __init__(
        self,
        laparams: Optional[LAParams] = None,
        style: bool = False,
    ):

        self.laparams = laparams or LAParams()
        self.style = style

    def generate_lines(self, pdf: bytes) -> Optional[pd.DataFrame]:
        """
        Generates dataframe from all blocs in the PDF.

        Arguments
        ---------
        pdf:
            Byte stream representing the PDF.

        Returns
        -------
        pd.DataFrame :
            DataFrame representing the blocs.
        """

        pdf_stream = BytesIO(pdf)

        layout = extract_pages(pdf_stream, laparams=self.laparams)
        lines = list(get_lines(layout))

        if not lines:
            return None

        df = pd.DataFrame.from_records([line.dict() for line in lines])

        return df

    def process(self, pdf: bytes) -> Optional[pd.DataFrame]:
        """
        Process a single PDF document.

        Parameters
        ----------
        pdf : bytes
            Raw byte representation of the PDF document.

        Returns
        -------
        Optional[pd.DataFrame]
            Dataframe containing one row for each line extracted using PDFMiner.
        """

        lines = self.generate_lines(pdf)

        if lines is None:
            return None

        lines = extract_text(lines, inplace=True)

        if self.style:
            lines = extract_styled_text(lines, inplace=True)

        lines.drop(columns=["line"], inplace=True)

        # Remove empty lines
        lines = lines[lines.text.str.len() > 0]

        # Remove lines that are outside the page
        lines = remove_outside_lines(lines, strict_mode=True)

        return lines

    def __call__(self, pdf: bytes) -> pd.DataFrame:
        return self.process(pdf)
