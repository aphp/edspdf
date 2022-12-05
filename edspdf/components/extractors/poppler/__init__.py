import warnings
from typing import Union

from edspdf import Component, registry
from edspdf.models import PDFDoc, SpannedStyle, TextBox

try:
    from .bindings import Document
except ModuleNotFoundError:
    warnings.warn(
        "Poppler was not correctly built, you won't be able to use the "
        "poppler.v1 component to parse PDFs."
    )


@registry.factory.register("poppler")
class PopplerExtractor(Component[Union[str, PDFDoc], PDFDoc]):
    """
    Extractor object. Given a PDF byte stream, produces a list of elements.

    Parameters
    ----------
    extract_style : bool
        Extract style
    """

    def __init__(
        self,
        extract_style: bool = False,
    ):
        super().__init__()

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

        if isinstance(doc, bytes):
            content = doc
            doc = PDFDoc(id=str(hash(content)), content=content)
        else:
            content = doc.content

        poppler_doc = Document(content, extract_style=self.extract_style)
        lines = []
        for page_no, page in enumerate(poppler_doc):
            w, h = page.size

            for flow in page:
                for block in flow:
                    for line in block:
                        if len(line.text) == 0:
                            continue
                        if self.extract_style:
                            styles = [
                                SpannedStyle(
                                    fontname=style.font_name,
                                    italic=style.is_italic,
                                    bold=style.is_bold,
                                    begin=style.begin,
                                    end=style.end,
                                )
                                for style in line.styles
                            ]
                        else:
                            styles = []
                        lines.append(
                            TextBox(
                                x0=line.x0 / w,
                                y0=line.y0 / h,
                                x1=line.x1 / w,
                                y1=line.y1 / h,
                                page_width=w,
                                page_height=h,
                                text=line.text,
                                page=page_no,
                                styles=tuple(styles),
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
