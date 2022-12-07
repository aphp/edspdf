import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTTextLineHorizontal

from edspdf import Component, registry
from edspdf.models import PDFDoc, SpannedStyle, TextBox

SPACE_PATTERN = re.compile(r"\s")
MULTISPACE_PATTERN = re.compile(r"\s+")


@registry.factory.register("pdfminer-extractor")
class PdfMinerExtractor(Component):
    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.05,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: Optional[float] = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False,
        extract_style: bool = False,
    ):
        """
        Extractor object. Given a PDF byte stream, produces a list of elements.

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
        extract_style : bool
            Whether to extract style (font, size, ...) information for each line of
            the document.
            Default: False
        """

        super().__init__()

        self.laparams = LAParams(
            line_overlap=line_overlap,
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            boxes_flow=boxes_flow,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )
        self.extract_style = extract_style

    def __call__(self, doc: Union[PDFDoc, bytes]) -> PDFDoc:
        """
        Extract blocks from a PDF from all blocks in the PDF.

        Arguments
        ---------
        doc:
            PDF document

        Returns
        -------
        PDFDoc:
            PDF document
        """

        if not isinstance(doc, PDFDoc):
            content = bytes(doc)
            doc = PDFDoc(id=str(hash(content)), content=content)
        content = doc.content
        content_stream = BytesIO(content)

        layout = extract_pages(content_stream, laparams=self.laparams)
        lines = []

        page_count = 0
        for page_no, page in enumerate(layout):

            page_count += 1

            w = page.width
            h = page.height

            for bloc in page:
                if not isinstance(bloc, LTTextBoxHorizontal):
                    continue
                bloc: LTTextBoxHorizontal

                for line in bloc:
                    text, styles = extract_style_from_line(line)
                    if len(text) == 0:
                        continue
                    lines.append(
                        TextBox(
                            page=page_no,
                            x0=line.x0 / w,
                            x1=line.x1 / w,
                            y0=1 - line.y1 / h,
                            y1=1 - line.y0 / h,
                            page_width=w,
                            page_height=h,
                            text=text,
                            styles=styles if self.extract_style else (),
                        )
                    )

        doc.lines = sorted(
            [
                line
                for line in lines
                if line.x0 >= 0 and line.y0 >= 0 and line.x1 <= 1 and line.y1 <= 1
            ]
        )

        return doc


def extract_style_from_line(
    line: LTTextLineHorizontal,
) -> Tuple[str, List[SpannedStyle]]:
    styles = []

    current_style = None
    text = ""
    last = None

    line_chars = []

    for char in line:

        new_char_text = SPACE_PATTERN.sub(" ", char._text)
        # We do not allow double spaces or spaces at the beginning of a text
        if not (new_char_text == " " and (text.endswith(" ") or text == "")):
            new_text = text + new_char_text
        else:
            new_text = text

        line_chars.append(new_char_text)
        line_chars.append(char)

        if new_char_text == " ":
            if last is not None:
                fontname, italic, bold = last
            else:
                fontname, italic, bold = (None, None, None)
        else:
            fontname = getattr(char, "fontname", "")
            italic = not char.upright or "italic" in char.fontname.lower()
            bold = "bold" in char.fontname.lower()

        if (fontname, italic, bold) != last:
            if current_style is not None:
                styles.append(current_style)
            current_style = SpannedStyle(
                fontname=fontname,
                italic=italic,
                bold=bold,
                begin=len(text),
                end=len(new_text),
            )

        elif new_char_text != " ":
            current_style.end = len(new_text)
        text = new_text
        last = (fontname, italic, bold)

    if current_style is not None:
        styles.append(current_style)

    # Remove spaces at the end of a text
    return text.rstrip(), tuple(styles)
