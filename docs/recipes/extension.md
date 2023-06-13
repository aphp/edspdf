# Extending EDS-PDF

EDS-PDF is organised around a function registry powered by catalogue and a custom configuration system. The result is a powerful framework that is easy to extend - and we'll see how in this section.

For this recipe, let's imagine we're not entirely satisfied with the aggregation
proposed by EDS-PDF. For instance, we might want an aggregator that outputs the
text in Markdown format.

!!! note

    Properly converting to markdown is no easy task. For this example,
    we will limit ourselves to detecting bold and italics sections.

## Developing the new aggregator

Our aggregator will inherit from the [`SimpleAggregator`][edspdf.pipes.aggregators.simple.SimpleAggregator],
and use the style to detect italics and bold sections.

```python title="markdown_aggregator.py"
from edspdf import registry
from edspdf.pipes.aggregators.simple import SimpleAggregator
from edspdf.structures import PDFDoc, Text


@registry.factory.register("markdown-aggregator")  # (1)
class MarkdownAggregator(SimpleAggregator):
    def __call__(self, doc: PDFDoc) -> PDFDoc:
        doc = super().__call__(doc)

        for label in doc.aggregated_texts.keys():
            text = doc.aggregated_texts[label].text

            fragments = []

            offset = 0
            for s in doc.aggregated_texts[label].properties:
                if s.begin >= s.end:
                    continue
                if offset < s.begin:
                    fragments.append(text[offset : s.begin])

                offset = s.end
                snippet = text[s.begin : s.end]
                if s.bold:
                    snippet = f"**{snippet}**"
                if s.italic:
                    snippet = f"_{snippet}_"
                fragments.append(snippet)

            if offset < len(text):
                fragments.append(text[offset:])

            doc.aggregated_texts[label] = Text(text="".join(fragments))

        return doc
```

1. The new aggregator is registered via this line
2. The new aggregator redefines the `__call__` method.
   It will output a single string, corresponding to the markdown-formatted output.

That's it! You can use this new aggregator with the API:

```python
from edspdf import Pipeline
from markdown_aggregator import MarkdownAggregator  # (1)

model = Pipeline()
# will extract text lines from a document
model.add_pipe(
    "pdfminer-extractor",
    config=dict(
        extract_style=False,
    ),
)
# classify everything inside the `body` bounding box as `body`
model.add_pipe("mask-classifier", config={"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9})
# aggregates the lines together to generate the markdown formatted text
model.add_pipe("markdown-aggregator")
```

1. We're importing the aggregator that we just defined.

It all works relatively smoothly!

## Making the aggregator discoverable

Now, how can we instantiate the pipeline using the configuration system?
The registry needs to be aware of the new function, but we shouldn't have to
import `mardown_aggregator.py` just so that the module is registered as a side-effect...

Catalogue solves this problem by using Python _entry points_.

=== "pyproject.toml"

    ```toml
    [project.entry-points."edspdf_factories"]
    "markdown-aggregator" = "markdown_aggregator:MarkdownAggregator"
    ```

=== "setup.py"

    ```python
    from setuptools import setup

    setup(
        name="edspdf-markdown-aggregator",
        entry_points={
            "edspdf_factories": [
                "markdown-aggregator = markdown_aggregator:MarkdownAggregator"
            ]
        },
    )
    ```

By declaring the new aggregator as an entrypoint, it will become discoverable by EDS-PDF
as long as it is installed in your environment!
