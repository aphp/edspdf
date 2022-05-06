from typing import Optional

from pdfminer.layout import LAParams

from edspdf.extraction.extractor import LineExtractor
from edspdf.utils.registry import registry


@registry.params.register("laparams.v1")
def laparams_factory(
    line_overlap: float = 0.5,
    char_margin: float = 2.0,
    line_margin: float = 0.5,
    word_margin: float = 0.1,
    boxes_flow: Optional[float] = 0.5,
    detect_vertical: bool = False,
    all_texts: bool = False,
):

    return LAParams(
        line_overlap=line_overlap,
        char_margin=char_margin,
        line_margin=line_margin,
        word_margin=word_margin,
        boxes_flow=boxes_flow,
        detect_vertical=detect_vertical,
        all_texts=all_texts,
    )


@registry.extractors.register("line-extractor.v1")
def extractor_factory(
    laparams: LAParams,
    style: bool,
):
    return LineExtractor(laparams=laparams, style=style)
