import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTTextLineHorizontal
from pdfminer.pdftypes import PDFException

from edspdf import Page, PDFDoc, Pipeline, TextBox, TextProperties, registry

SPACE_PATTERN = re.compile(r"\s")
MULTISPACE_PATTERN = re.compile(r"\s+")


@registry.factory.register("pdfminer_extractor")
class PdfMinerExtractor:
    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        name: str = "pdfminer_extractor",
        line_overlap: float = 0.5,
        char_margin: float = 2.05,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: Optional[float] = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False,
        extract_properties: bool = False,
        raise_on_error: bool = False,
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
        extract_properties : bool
            Whether to extract style (font, size, ...) information for each line of
            the document.
            Default: False
        """

        self.name = name

        self.laparams = LAParams(
            line_overlap=line_overlap,
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            boxes_flow=boxes_flow,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )
        self.extract_properties = extract_properties
        self.raise_on_error = raise_on_error

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
            doc = PDFDoc(
                id=str(hash(content)),
                content=content,
            )
        content = doc.content
        content_stream = BytesIO(content)
        pages = doc.pages = []

        try:
            layout = list(extract_pages(content_stream, laparams=self.laparams))
        except PDFException:
            if self.raise_on_error:
                raise
            doc.pages = []
            doc.error = True
            return doc

        doc.content_boxes = []
        page_count = 0
        for page_no, pdfminer_page in enumerate(layout):
            page_count += 1

            w = pdfminer_page.width
            h = pdfminer_page.height

            content_boxes = []
            page = Page(
                doc=doc,
                page_num=page_no,
                width=w,
                height=h,
            )
            pages.append(page)

            for bloc in pdfminer_page:
                if not isinstance(bloc, LTTextBoxHorizontal):
                    continue
                bloc: LTTextBoxHorizontal

                for line in bloc:
                    text, props = extract_properties_from_line(line)  # type: ignore
                    if len(text) == 0:
                        continue
                    content_boxes.append(
                        TextBox(
                            doc=doc,
                            page_num=page_no,
                            x0=line.x0 / w,
                            x1=line.x1 / w,
                            y0=1 - line.y1 / h,
                            y1=1 - line.y0 / h,
                            text=text,
                            props=props if self.extract_properties else (),
                        )
                    )

            doc.content_boxes.extend(
                sorted(
                    [
                        box
                        for box in content_boxes
                        if (box.x0 >= 0 and box.y0 >= 0 and box.x1 <= 1 and box.y1 <= 1)
                    ]
                )
            )

        return doc


def extract_properties_from_line(
    line: LTTextLineHorizontal,
) -> Tuple[str, List[TextProperties]]:
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
            current_style = TextProperties(
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
    return text.rstrip(), styles
