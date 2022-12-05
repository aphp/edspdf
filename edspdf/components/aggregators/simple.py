from itertools import groupby
from typing import Dict, List

import numpy as np

from edspdf import Component, registry
from edspdf.models import PDFDoc, TextBox


@registry.factory.register("simple-aggregator")
class SimpleAggregator(Component):
    def __init__(
        self,
        new_line_threshold: float = 0.2,
        new_paragraph_threshold: float = 1.5,
        label_map: Dict = {},
    ) -> None:

        super().__init__()
        self.label_map = dict(label_map)
        self.new_line_threshold = new_line_threshold
        self.new_paragraph_threshold = new_paragraph_threshold

    def __call__(self, doc: PDFDoc) -> Dict[str, str]:

        row_height = sum(b.y1 - b.y0 for b in doc.lines) / len(doc.lines)
        all_lines = sorted(
            [
                line
                for line in doc.lines
                if len(line.text) > 0 and line.label is not None
            ],
            key=lambda b: (b.label, b.page, b.y1 // row_height, b.x0),
        )

        texts = {}
        for label, lines in groupby(all_lines, key=lambda b: b.label):
            text_parts = []
            lines: List[TextBox] = list(lines)
            pairs = list(zip(lines, [*lines[1:], None]))
            dys = [
                next_box.y1 - line.y1
                if next_box is not None and line.page == next_box.page
                else None
                for line, next_box in pairs
            ]
            height = np.median(np.asarray([line.y1 - line.y0 for line in lines]))
            for (line, next_box), dy in zip(pairs, dys):
                text_parts.append(line.text)
                if next_box is None:
                    continue
                if line.page != next_box.page:
                    text_parts.append("\n\n")
                elif dy / height > self.new_paragraph_threshold:
                    text_parts.append("\n\n")
                elif dy / height > self.new_line_threshold:
                    text_parts.append("\n")
                else:
                    text_parts.append(" ")
            texts[label] = "".join(text_parts)

        return texts
