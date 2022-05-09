from typing import Dict

import pandas as pd

from edspdf.reg import registry

from .base import BaseAggregator


@registry.aggregators.register("simple.v1")
class SimpleAggregator(BaseAggregator):
    def __init__(
        self,
        new_line_threshold: float = 0.2,
        new_paragraph_threshold: float = 1.2,
    ) -> None:

        self.nl_threshold = new_line_threshold
        self.np_threshold = new_paragraph_threshold

    def prepare_newlines(self, lines: pd.DataFrame) -> pd.DataFrame:

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
            "\n\n",
        )

        lines["text_with_newline"] = lines.text + lines.newline

        return lines

    def aggregate(self, lines: pd.DataFrame) -> Dict[str, str]:

        lines = self.prepare_newlines(lines)

        df = lines.groupby(["prediction"]).agg(text=("text_with_newline", "sum"))

        return df.text.to_dict()
