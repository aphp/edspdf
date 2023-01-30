# Pipelines overview

EDS-PDF provides easy-to-use components for defining PDF processing pipelines.

=== "Box extractors"

    | Pipeline             | Description                                     |
    |----------------------|-------------------------------------------------|
    | `pdfminer-extractor` | Extracts text lines with the `pdfminer` library |
    | `mupdf-extractor`    | Extracts text lines with the `pymupdf` library  |
    | `poppler-extractor`  | Extracts text lines with the `poppler` software |

=== "Box classifiers"

    | Pipeline                | Description                             |
    |-------------------------|-----------------------------------------|
    | `deep-classifier`       | Trainable box classification model      |
    | `mask-classifier`       | Simple rule-based classification        |
    | `multi-mask-classifier` | Simple rule-based classification        |
    | `dummy-classifier`      | Dummy classifier, for testing purposes. |
    | `random-classifier`     | To sow chaos                            |


=== "Aggregators"

    | Method              | Description                                                       |
    |---------------------|-------------------------------------------------------------------|
    | `simple-aggregator` | Returns a dictionary with one key for each detected class         |
    | `styled-aggregator` | Returns the same dictionary, as well as the information on styles |

You can add them to your EDS-PDF pipeline by simply calling `add_pipe`, for instance:

<!-- no-check -->

```python
# ↑ Omitted code that defines the pipeline object ↑
pipeline.add_pipe("pdfminer-extractor", name="component-name", config=...)
```
