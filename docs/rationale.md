# Structure and Rationale

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

## Modularity

EDS-PDF includes everything you need to get started on text extraction, and ships with a number
of trainable classifiers. But it also makes it extremely easy to extend its functionalities by
designing new pipelines.

Following the example of spaCy, EDS-PDF is organised around Explosion's
[`catalogue` library](https://github.com/explosion/catalogue), enabling a powerful configuration
system based on an extendable registry.

Much like spaCy pipelines, EDS-PDF pipelines are meant to be reproducible and serialisable,
such that the primary way to define a pipeline is through the configuration system.

For more information on the configuration system, refer to the documentations of
[Thinc](https://thinc.ai/docs/usage-config) and [spaCy](https://spacy.io/usage/training#config).
