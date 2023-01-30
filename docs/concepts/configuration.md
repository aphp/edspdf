# Configuration

Following the example of spaCy, EDS-PDF is organised a powerful configuration system and registries organised with Explosion's
[`catalogue`](https://github.com/explosion/catalogue) library.

The following catalogues are included within EDS-PDF:

| Section       | Description                               |
|---------------|-------------------------------------------|
| `factory`     | Components factories (most often classes) |
| `adapter`     | Raw data preprocessing functions          |

Much like spaCy pipelines, EDS-PDF pipelines are meant to be reproducible and serialisable, such that you can always define a pipeline through the configuration system.

To wit, compare the API-based approach to the configuration-based approach (the two are strictly equivalent):

=== "API-based"

    ```python hl_lines="4-13"
    from edspdf import aggregation, reading, extraction, classification
    from pathlib import Path

    model = spacy.Pipeline()
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

    texts = model(pdf)

    texts["body"]
    # Out: Cher Pr ABC, Cher DEF,\n...
    ```

=== "Configuration-based"

    ```toml title="config.cfg"
    [pipeline]
    components = ["extractor", "classifier", "aggregator"]
    components_config = ${components}

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
    @aggregators = "simple-extractor"
    ```

    ```python hl_lines="4-5"
    from edspdf import registry, Config
    from pathlib import Path

    config = Config().from_disk("config.cfg")
    pipeline = config.resolve()["pipeline"]

    # Get a PDF
    pdf = Path("letter.pdf").read_bytes()

    texts = pipeline(pdf)

    texts["body"]
    # Out: Cher Pr ABC, Cher DEF,\n...
    ```

The configuration-based approach strictly separates the definition of the pipeline
to its application and avoids tucking away important configuration details.
Changes to the pipeline are transparent as there is a single source of truth: the configuration file.

## Interpolation

Our configuration system relies heavily on interpolation to make it easier to define complex architectures within a single configuration file. However, unlike [confection](https://github.com/explosion/confection), EDS-PDF replaces interpolated variables after they are resolved using registries, which allows sharing model parts between components when defining pipelines.
