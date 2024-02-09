import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import pypdfium2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTTextLineHorizontal
from pdfminer.pdftypes import PDFException

from edspdf import Page, PDFDoc, Pipeline, TextBox, TextProperties, registry


@registry.factory.register("pdfminer-extractor")
class PdfMinerExtractor:
    """
    We provide a PDF line extractor built on top of
    [PdfMiner](https://pdfminersix.readthedocs.io/en/latest/).

    This is the most portable extractor, since it is pure-python and can therefore
    be run on any platform. Be sure to have a look at their documentation,
    especially the [part providing a bird's eye view of the PDF extraction process](https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html).

    Examples
    --------

    === "API-based"

        ```python
        pipeline.add_pipe(
            "pdfminer-extractor",
            config=dict(
                extract_style=False,
            ),
        )
        ```

    === "Configuration-based"

        ```toml
        [components.extractor]
        @factory = "pdfminer-extractor"
        extract_style = false
        ```

    And use the pipeline on a PDF document:

    ```python
    from pathlib import Path

    # Apply on a new document
    pipeline(Path("path/to/your/pdf/document").read_bytes())
    ```

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
    render_pages: bool
        Whether to extract the rendered page as a numpy array in the `page.image`
        attribute (defaults to False)
    render_dpi: int
        DPI to use when rendering the page (defaults to 200)
    raise_on_error : bool
        Whether to raise an error if the PDF cannot be parsed.
        Default: False
    """  # noqa: E501

    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        name: str = "pdfminer-extractor",
        line_overlap: float = 0.5,
        char_margin: float = 2.05,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: Optional[float] = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False,
        extract_style: bool = False,
        raise_on_error: bool = False,
        render_pages: bool = False,
        render_dpi: int = 200,
    ):
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
        self.extract_style = extract_style
        self.raise_on_error = raise_on_error
        self.render_pages = render_pages
        self.render_dpi = render_dpi

    def __call__(self, doc: Union[PDFDoc, bytes]) -> PDFDoc:
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
                            props=props if self.extract_style else (),
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

        if self.render_pages:
            # See https://pypdfium2.readthedocs.io/en/stable/python_api.html#user-unit
            pdfium_doc = pypdfium2.PdfDocument(content)
            for page, pdfium_page in zip(pages, pdfium_doc):
                image = pdfium_page.render(scale=self.render_dpi / 72).to_pil()
                np_img = np.array(image)
                page.image = np_img

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
        new_char_text = re.sub(r"\s", " ", char._text)
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
