from io import BytesIO
from typing import Optional

import pandas as pd
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams

from edspdf.reg import registry

from .base import BaseExtractor
from .functional import get_lines, remove_outside_lines


@registry.extractors.register("pdfminer.v1")
class PdfMinerExtractor(BaseExtractor):
    """
    Extractor object. Given a PDF byte stream, produces a list of blocs.

    Parameters
    ----------
    line_overlap : float
        See PDFMiner documentation
    char_margin : float
        See PDFMiner documentation
    line_margin : float
        See PDFMiner documentation
    word_margin : float
        See PDFMiner documentation
    boxes_flow : Optional[float]
        See PDFMiner documentation
    detect_vertical : bool
        See PDFMiner documentation
    all_texts : bool
        See PDFMiner documentation
    """

    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: Optional[float] = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False,
    ):

        self.laparams = LAParams(
            line_overlap=line_overlap,
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            boxes_flow=boxes_flow,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )

    def generate_lines(self, pdf: bytes) -> pd.DataFrame:
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
            return pd.DataFrame(
                columns=[
                    "page",
                    "bloc",
                    "x0",
                    "x1",
                    "y0",
                    "y1",
                    "page_width",
                    "page_height",
                    "text",
                    "styles",
                ]
            )

        df = pd.DataFrame.from_records([line.dict() for line in lines])
        df["line_id"] = range(len(df))

        return df

    def extract(self, pdf: bytes) -> pd.DataFrame:
        """
        Process a single PDF document.

        Parameters
        ----------
        pdf : bytes
            Raw byte representation of the PDF document.

        Returns
        -------
        pd.DataFrame
            DataFrame containing one row for each line extracted using PDFMiner.
        """

        lines = self.generate_lines(pdf)

        # Remove empty lines
        lines = lines[lines.text.str.len() > 0]

        # Remove lines that are outside the page
        lines = remove_outside_lines(lines, strict_mode=True)

        return lines
