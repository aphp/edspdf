from itertools import groupby
from typing import List

import numpy as np

from edspdf import registry
from edspdf.structures import PDFDoc, Text, TextBox

from .simple import SimpleAggregator


@registry.factory.register("styled_aggregator")
class StyledAggregator(SimpleAggregator):
    """
    Aggregator that returns text and styles.
    """

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        all_lines = doc.text_boxes
        row_height = sum(b.y1 - b.y0 for b in all_lines) / max(1, len(doc.text_boxes))
        all_lines = sorted(
            [
                line
                for line in all_lines
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
                for style in line.props:
                    styles[label].append(
                        style.evolve(
                            begin=style.begin + len(text),
                            end=style.end + len(text),
                        )
                    )
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
            texts[label] = Text(
                text="".join(text),
                properties=styles[label],
            )

        doc.aggregated_texts.update(texts)
        return doc
