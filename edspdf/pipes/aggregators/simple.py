from itertools import groupby
from typing import Dict, List

import numpy as np

from edspdf import PDFDoc, Pipeline, Text, TextBox, registry


@registry.factory.register("simple-aggregator")
class SimpleAggregator:
    """
    Aggregator that returns texts and styles. It groups all text boxes with the same
    label under the `aggregated_text`, and additionally aggregates the
    [styles][edspdf.structures.TextProperties] of the text boxes.

    Examples
    --------

    Create a pipeline

    === "API-based"

        ```python
        pipeline = ...
        pipeline.add_pipe(
            "simple-aggregator",
            name="aggregator",
            config={
                "new_line_threshold": 0.2,
                "new_paragraph_threshold": 1.5,
                "label_map": {
                    "body": "text",
                    "table": "text",
                },
            },
        )
        ```

    === "Configuration-based"

        ```toml
        ...

        [components.aggregator]
        @factory = "simple-aggregator"
        new_line_threshold = 0.2
        new_paragraph_threshold = 1.5
        label_map = { body = "text", table = "text" }

        ...
        ```

    and run it on a document:

    ```python
    doc = pipeline(doc)
    print(doc.aggregated_texts)
    # {
    #     "text": "This is the body of the document, followed by a table | A | B |"
    # }
    ```

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline object
    name: str
        The name of the component
    sort: bool
        Whether to sort text boxes inside each label group by (page, y, x) position
        before merging them.
    new_line_threshold: float
        Minimum ratio of the distance between two lines to the median height of
        lines to consider them as being on separate lines
    new_paragraph_threshold: float
        Minimum ratio of the distance between two lines to the median height of
        lines to consider them as being on separate paragraphs and thus add a
        newline character between them.
    label_map: Dict
        A dictionary mapping labels to new labels. This is useful to group labels
        together, for instance, to output both "body" and "table" as "text".
    """

    def __init__(
        self,
        pipeline: Pipeline = None,
        name: str = "simple-aggregator",
        sort: bool = False,
        new_line_threshold: float = 0.2,
        new_paragraph_threshold: float = 1.5,
        label_map: Dict = {},
    ) -> None:
        self.name = name
        self.sort = sort
        self.label_map = dict(label_map)
        self.new_line_threshold = new_line_threshold
        self.new_paragraph_threshold = new_paragraph_threshold

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        all_lines = doc.text_boxes
        row_height = sum(b.y1 - b.y0 for b in all_lines) / max(1, len(doc.text_boxes))
        all_lines = [
            line for line in all_lines if len(line.text) > 0 and line.label is not None
        ]
        if self.sort:
            all_lines = sorted(
                all_lines,
                key=lambda b: (b.label, b.page_num, b.y1 // row_height, b.x0),
            )
        else:
            all_lines = sorted(all_lines, key=lambda b: b.label)

        texts = {}
        styles = {}
        for label, lines in groupby(all_lines, key=lambda b: b.label):
            styles[label] = []
            text = ""
            lines: List[TextBox] = list(lines)
            pairs = list(zip(lines, [*lines[1:], None]))
            dys = [
                next_box.y1 - line.y1
                if next_box is not None and line.page_num == next_box.page_num
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
                if line.page_num != next_box.page_num:
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
