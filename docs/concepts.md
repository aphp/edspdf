# Key Concepts

The goal of EDS-PDF is to provide a **framework** for text extraction from PDF documents,
along with some utilities and a few pipelines, stitched together by a robust configuration
system powered by [Thinc](https://thinc.ai/docs/usage-config).

## Organisation

The core object within EDS-PDF is the `reader`, which organises the extraction along four
well-defined steps:

1. The **extraction** step extracts text blocs from the PDF and compiles them into a pandas DataFrame
   object, where each row relates to a single bloc.
2. The **transformation** step is optional. It computes user-defined transformation on the data,
   to provide the classification algorithm with additional features.
3. The **classification** step categorises each bloc, typically between `body`, `header`, `footer`...
4. The **aggregation** step compiles the blocs together, exploiting the classification to re-create the original text.

## Data Structure

EDS-PDF parses the PDF into a pandas DataFrame object where each row represents a text bloc.
The DataFrame is carried all the way down to the aggregation step.

The following columns are reserved:

| Column  | Description                                                             |
| ------- | ----------------------------------------------------------------------- |
| `text`  | Bloc text content                                                       |
| `page`  | Page within the PDF (starting at 0)                                     |
| `x0`    | Horizontal position of the top-left corner of the bloc bounding box     |
| `x1`    | Horizontal position of the bottom-right corner of the bloc bounding box |
| `y0`    | Vertical position of the top-left corner of the bloc bounding box       |
| `y1`    | Vertical position of the bottom-right corner of the bloc bounding box   |
| `label` | Class of the bloc (eg `body`, `header`...)                              |

!!! note "Position of bloc bounding boxes"

    The positional information (columns `x0/x1/y0/y1`) is normalised, and takes the top-left corner of
    the page as reference.

    Note that this contrasts with the PDF convention, which takes the **bottom left corner** as origin instead.

Some transformations may create their own columns. It's your responsibility to verify that
the column names do not override each other.

We can review the different stages of the pipeline:

| Step           | Input       | Output    | Description                                       |
| -------------- | ----------- | --------- | ------------------------------------------------- |
| Extraction     | PDF (bytes) | DataFrame | Extracts text blocs from the PDF                  |
| Transformation | DataFrame   | DataFrame | Compute hand-defined transformations on the blocs |
| Classification | DataFrame   | DataFrame | Categorises each bloc                             |
| Aggregation    | DataFrame   | Dict      | Re-creates the original text                      |

## Configuration

Following the example of spaCy, EDS-PDF is organised around Explosion's
[`catalogue` library](https://github.com/explosion/catalogue), enabling a powerful configuration
system based on an extendable registry.

The following catalogues are included within EDS-PDF:

| Section       | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `readers`     | Top-level object, encapsulating a full EDS-PDF pipeline                |
| `extractors`  | Text bloc extraction models                                            |
| `transforms`  | Transformations that can be applied to each bloc before classification |
| `classifiers` | Classification routines (eg rule- or ml-based)                         |
| `misc`        | Some miscellaneous utility functions                                   |

Much like spaCy pipelines, EDS-PDF pipelines are meant to be reproducible and serialisable,
such that the primary way to define a pipeline is through the configuration system.

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

For more information on the configuration system, refer to the documentations of
[Thinc](https://thinc.ai/docs/usage-config) and [spaCy](https://spacy.io/usage/training#config).

## Modularity and Extensibility

EDS-PDF includes everything you need to get started on text extraction, and ships with a number
of trainable classifiers. But it also makes it extremely easy to extend its functionalities by
designing new pipelines.
