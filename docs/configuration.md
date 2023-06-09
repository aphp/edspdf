# Configuration

EDS-PDF is built on top of the [`confit`](https://github.com/aphp/confit) configuration system.

The following [catalogue](https://github.com/explosion/catalogue) registries are included within EDS-PDF:

| Section       | Description                               |
|---------------|-------------------------------------------|
| `factory`     | Components factories (most often classes) |
| `adapter`     | Raw data preprocessing functions          |

EDS-PDF pipelines are meant to be reproducible and serializable, such that you can always define a pipeline through the configuration system.

To wit, compare the API-based approach to the configuration-based approach (the two are strictly equivalent):

=== "API-based"

    ```python hl_lines="4-13"
    import edspdf
    from pathlib import Path

    model = edspdf.Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe("mask-classifier", name="classifier", config=dict(
        x0=0.2,
        x1=0.9,
        y0=0.3,
        y1=0.6,
        threshold=0.1,
    )
    model.add_pipe("simple-aggregator", name="aggregator")

    # Get a PDF
    pdf = Path("letter.pdf").read_bytes()

    pdf = model(pdf)

    str(pdf.aggregated_texts["body"])
    # Out: Cher Pr ABC, Cher DEF,\n...
    ```

=== "Configuration-based"

    ```toml title="config.cfg"
    [pipeline]
    pipeline = ["extractor", "classifier", "aggregator"]

    [components.extractor]
    @factory = "pdfminer-extractor"

    [components.classifier]
    @factory = "mask-classifier"
    x0 = 0.2
    x1 = 0.9
    y0 = 0.3
    y1 = 0.6
    threshold = 0.1

    [components.aggregator]
    @factory = "simple-aggregator"
    ```

    ```python hl_lines="4"
    import edspdf
    from pathlib import Path

    pipeline = edspdf.load("config.cfg")

    # Get a PDF
    pdf = Path("letter.pdf").read_bytes()

    pdf = pipeline(pdf)

    str(pdf.aggregated_texts["body"])
    # Out: Cher Pr ABC, Cher DEF,\n...
    ```

The configuration-based approach strictly separates the definition of the pipeline
to its application and avoids tucking away important configuration details.
Changes to the pipeline are transparent as there is a single source of truth: the configuration file.
