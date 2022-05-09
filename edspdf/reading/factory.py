from typing import Callable, Optional

import pandas as pd

from edspdf.reading.reader import Classifier, LineExtractor, PdfReader
from edspdf.reg import registry


@registry.readers.register("pdf-reader.v1")
def pdfreader_factory(
    extractor: LineExtractor,
    classifier: Classifier,
    transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    new_line_threshold: float = 0.2,
    new_paragraph_threshold: float = 1.2,
):
    return PdfReader(
        extractor=extractor,
        classifier=classifier,
        transform=transform,
        new_line_threshold=new_line_threshold,
        new_paragraph_threshold=new_paragraph_threshold,
    )
