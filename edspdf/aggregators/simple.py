from typing import Dict, Optional

import pandas as pd

from edspdf.reg import registry

from .base import BaseAggregator
from .functional import prepare_newlines


@registry.aggregators.register("simple.v1")
class SimpleAggregator(BaseAggregator):
    def __init__(
        self,
        new_line_threshold: float = 0.2,
        new_paragraph_threshold: float = 1.5,
        label_map: Optional[Dict] = None,
    ) -> None:

        self.nl_threshold = new_line_threshold
        self.np_threshold = new_paragraph_threshold
        self.label_map = {} if label_map is None else label_map

    def aggregate(self, lines: pd.DataFrame) -> Dict[str, str]:

        if len(lines) == 0:
            return {}

        lines = lines.sort_values(["page", "y1", "x0"])
        lines["label"] = lines["label"].map(lambda l: self.label_map.get(l, l))

        lines = prepare_newlines(
            lines,
            nl_threshold=self.nl_threshold,
            np_threshold=self.np_threshold,
        )

        df = lines.groupby(["label"]).agg(text=("text_with_newline", "sum"))

        return df.text.to_dict()
