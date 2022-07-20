# Extending EDS-PDF

EDS-PDF is organised around a function registry powered by catalogue,
which plays extremely well with Thinc's configuration system.
The result is a powerful framework that is easy to extend - and we'll
see how in this section.

For this recipe, let's imagine we're not entirely satisfied with the aggregation
proposed by EDS-PDF. For instance, we might want an aggregator that outputs the
text in markdown format.

!!! note

    Properly converting to markdown is no easy task. For this example,
    we will limit ourselves to detecting italics sections.

## Developing the new aggregator

Our aggregator will inherit from the [`StyledAggregator`][edspdf.aggregation.styled.StyledAggregator],
and use the style to detect italics and bold sections.

```python title="markdown_aggregator.py"
from edspdf import registry
from typing import Any, Dict

from edspdf.aggregation.styled import StyledAggregator


def process_style(style: Dict[str, Any]) -> bool, bool, int, int:
    bold = "bold" in style["style"].lower()
    italics = (not style["upright"]) or ("italics" in style["style"].lower())

    return bold, italics, style["start"], style["end"]


@registry.aggregators.register("markdown.v1")  # (1)
class MarkdownAggregator(SimpleAggregator):
    def aggregate(self, lines: pd.DataFrame) -> str:  # (2)

        texts, styles = super(self).aggregate(lines)

        body = texts.get("body", "")
        style = styles.get("body", [])

        fragments = []

        for s in style:
            bold, italics, start, end = process_style(s)

            text = body[start:end]

            if bold:
                text = f"**{text}**"
            if italics:
                text = f"_{text}_"

            fragments.append(text)

        return "".join(fragments)
```

1. The new aggregator is registered via this line
2. The new aggregator redefines the `aggregate` method.
   It will output a single string, corresponding to the markdown-formatted output.

That's it! You can use this new aggregator with the API:

```python
from edspdf import reading, extraction, classification
from markdown_aggregator import MarkdownAggregator  # (1)
from pathlib import Path

reader = reading.PdfReader(
    extractor=extraction.PdfMinerExtractor(),
    classifier=classification.simple_mask_classifier_factory(
        x0=0.2,
        x1=0.9,
        y0=0.3,
        y1=0.6,
        threshold=0.1,
    ),
    aggregator=MarkdownAggregator(),
)
```

1. We're importing the aggregator that we just defined.

It all works relatively smoothly!

## Making the aggregator discoverable

Now, how can we instantiate the pipeline using the configuration system?
The registry needs to be aware of the new function, but we shouldn't have to
import `mardown_aggregator.py` just so that the module is registered as a side-effect...

Catalogue solves this problem by using Python _entry points_.

=== "pyproject.toml (Poetry)"

    ```toml
    [tool.poetry.plugins."edspdf_aggregators"]
    "custom.markdown.v1" = "markdown_aggregator:MarkdownAggregator"
    ```

=== "setup.py"

    ```python
    from setuptools import setup

    setup(
        name="edspdf-markdown",
        entry_points={
            "edspdf_aggregators": [
                "custom.markdown.v1 = markdown_aggregator:MarkdownAggregator"
            ]
        },
    )
    ```

By declaring the new aggregator as an entrypoint, it will become discoverable by EDS-PDF
as long as it is installed in your environment!
