from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np

from edspdf import registry
from edspdf.models import PDFDoc, TextBox

from .simple import SimpleAggregator


@registry.factory.register("styled-aggregator")
class StyledAggregator(SimpleAggregator):
    """
    Aggregator that returns text and styles.
    """

    def __call__(self, doc: PDFDoc) -> Tuple[Dict[str, str], Dict[str, List[Dict]]]:

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
        styles = {}
        for label, lines in groupby(all_lines, key=lambda b: b.label):
            styles[label] = []
            text = ""
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
                for style in line.styles:
                    style_dict = style.dict()
                    style_dict["begin"] += len(text)
                    style_dict["end"] += len(text)
                    styles[label].append(style_dict)
                text = text + line.text
                if next_box is None:
                    continue
                if line.page != next_box.page:
                    text = text + "\n\n"
                elif dy / height > self.new_paragraph_threshold:
                    text = text + "\n\n"
                elif dy / height > self.new_line_threshold:
                    text = text + "\n"
                else:
                    text = text + " "
            texts[label] = "".join(text)

        return texts, styles
