from typing import Callable, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from edspdf.extraction.extractor import LineExtractor

Classifier = Pipeline


class PdfReader:
    def __init__(
        self,
        extractor: LineExtractor,
        classifier: Pipeline,
        transform: Optional[Callable[[pd.DataFrame, bool], pd.DataFrame]] = None,
        new_line_threshold: float = 0.2,
        new_paragraph_threshold: float = 1.2,
    ) -> None:

        self.extractor = extractor
        self.classifier = classifier

        self.transform = transform

        self.nl_threshold = new_line_threshold
        self.np_threshold = new_paragraph_threshold

    def prepare_newlines(self, lines: pd.DataFrame, copy: bool = False) -> pd.DataFrame:

        if copy:
            lines = lines.copy()

        # Sort values before grouping
        lines = lines.sort_values(["page", "y1", "x0"])

        # Get information
        lines["next_y1"] = lines.groupby(["prediction"])["y1"].shift(-1)
        lines["next_page"] = lines.groupby(["prediction"])["page"].shift(-1)

        lines["dy"] = (lines.next_y1 - lines.y1).where(lines.next_y1 > lines.y1)
        lines["med_dy"] = lines.groupby(["prediction"])["dy"].transform("median")

        lines["newline"] = " "

        lines.newline = lines.newline.mask(
            lines.dy > lines.med_dy * self.nl_threshold,
            "\n",
        )
        lines.newline = lines.newline.mask(
            lines.dy > lines.med_dy * self.np_threshold,
            "\n\n",
        )

        lines.newline = lines.newline.mask(
            lines.page != lines.next_page,
            "\n",
        )

        lines["text_with_newline"] = lines.text + lines.newline

        return lines

    def predict(self, lines: pd.DataFrame, copy: bool = True) -> pd.DataFrame:

        if copy:
            lines = lines.copy()

        lines["prediction"] = self.classifier.predict(lines)
        lines["meta_prediction"] = lines.prediction.replace(
            {"section_title": "body", "table": "body"}
        )

        return lines

    def process_body(self, lines: pd.DataFrame) -> Optional[str]:

        df = lines.query("meta_prediction == 'body'")

        if len(df):
            body = "".join(df.text_with_newline)
            return body

        return None

    def process_admin(self, lines: pd.DataFrame) -> pd.DataFrame:

        df = lines.query("meta_prediction != 'body'")
        df = df.groupby(["prediction", "page"]).agg(text=("text_with_newline", "sum"))

        return df

    def process(self, pdf: bytes, orbis: bool) -> Optional[str]:

        lines = self.extractor(pdf)

        # Apply transformation
        if self.transform is not None:
            lines = self.transform(lines, orbis=orbis)

        lines = self.predict(lines)

        # We do not treat "pollution" lines for now
        lines = lines.query('meta_prediction == "body"')

        lines = self.prepare_newlines(lines)

        body = self.process_body(lines)

        return body

    def __call__(self, pdf: bytes, orbis: bool) -> str:
        return self.process(pdf, orbis)
