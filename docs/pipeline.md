# Pipeline

The library takes inspiration from spaCy pipelining system and closely follows its API.
Therefore, the core object within EDS-PDF is the Pipeline, which organises the processing of PDF documents
into multiple components.

A component is a processing block (like a function) that applies a transformation on its
input and returns a modified object.

At the moment, three types of components are implemented in the library:
1. **extraction** components extract lines from a raw PDF and return a `PDFDoc` object filled with these text boxes.
3. **classification** components classify each box, typically between `body`, `header`, `footer`...
4. **aggregation** components compiles the lines together according to their classes to re-create the original text.

To create your first pipeline, execute the following code:
To add a component (for instance, a `pdfminer` extraction component) to a pipeline, execute the following code :
```python
from edspdf import Pipeline

model = Pipeline()
# extract text lines from a document
model.add_pipe(
    "pdfminer",
    config=dict(
        extract_style=False,
    ),
)
# classify everything inside the `body` bounding box as `body`
model.add_pipe(
    "mask-classifier", config=dict(body={"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9})
)
# aggregate the lines together to re-create the original text
model.add_pipe("simple-aggregator")
```

This pipeline can then be run on one or more PDF documents.
As the pipeline process documents, components will be called in the order
they were added to the pipeline.

```python
from pathlib import Path

pdf_bytes = Path("path/to/your/pdf").read_bytes()

# Processing one document
model(pdf_bytes)

# Processing multiple documents
model.pipe([pdf_bytes, ...])
```

## Configuration

Following the example of spaCy, EDS-PDF is organised around Explosion's
[`catalogue` library](https://github.com/explosion/catalogue), enabling a powerful configuration
system based on an extendable registry.

The following catalogues are included within EDS-PDF:

| Section       | Description                               |
|---------------|-------------------------------------------|
| `factory`     | Components factories (most often classes) |
| `adapter`     | Raw data preprocessing functions          |

Much like spaCy pipelines, EDS-PDF pipelines are meant to be reproducible and serialisable,
such that you can always define a pipeline through the configuration system.

To wit, compare the API-based approach to the configuration-based approach (the two are strictly equivalent):

=== "API-based"

    ```python hl_lines="4-14"
    from edspdf import aggregation, reading, extraction, classification
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
        aggregator=aggregation.SimpleAggregation(),
    )

    # Get a PDF
    pdf = Path("letter.pdf").read_bytes()

    texts = reader(pdf)

    texts["body"]
    # Out: Cher Pr ABC, Cher DEF,\n...
    ```

=== "Configuration-based"

    ```toml title="config.cfg"
    [reader]
    @readers = "pdf-reader.v1"

    [reader.extractor]
    @extractors = "pdfminer.v1"

    [reader.classifier]
    @classifiers = "mask.v1"
    x0 = 0.2
    x1 = 0.9
    y0 = 0.3
    y1 = 0.6
    threshold = 0.1

    [reader.aggregator]
    @aggregators = "simple.v1"
    ```

    ```python hl_lines="4-5"
    from edspdf import registry, Config
    from pathlib import Path

    config = Config().from_disk("config.cfg")
    reader = registry.resolve(config)["reader"]

    # Get a PDF
    pdf = Path("letter.pdf").read_bytes()

    texts = reader(pdf)

    texts["body"]
    # Out: Cher Pr ABC, Cher DEF,\n...
    ```

The configuration-based approach strictly separates the definition of the pipeline
to its application and avoids tucking away important configuration details.
Changes to the pipeline are transparent as there is a single source of truth: the configuration file.
