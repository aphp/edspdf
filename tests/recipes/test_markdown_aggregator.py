from edspdf import registry
from edspdf.pipes.aggregators.simple import SimpleAggregator
from edspdf.structures import PDFDoc, Text


@registry.factory.register("markdown-aggregator")  #
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


def test_markdown_aggregator(styles_pdf):
    from edspdf import Pipeline

    model = Pipeline()
    # will extract text lines from a document
    model.add_pipe(
        "pdfminer-extractor",
        config=dict(
            extract_style=True,
        ),
    )
    # classify everything inside the `body` bounding box as `body`
    model.add_pipe(
        "mask-classifier",
        config={"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9},
    )
    # aggregates the lines together to re-create the original text
    model.add_pipe("markdown-aggregator")

    assert model(styles_pdf).aggregated_texts["body"].text == (
        "Let’s up the stakes, with _intra_-word change. Or better yet, **this mi**ght "
        "be hard."
    )
