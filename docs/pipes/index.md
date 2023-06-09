# Components overview

EDS-PDF provides easy-to-use components for defining PDF processing pipelines.



=== "Box extractors"

    --8<-- "docs/pipes/extractors/index.md:components"

=== "Box classifiers"

    --8<-- "docs/pipes/box-classifiers/index.md:components"


=== "Aggregators"

    --8<-- "docs/pipes/aggregators/index.md:components"


=== "Embeddings"

    --8<-- "docs/pipes/embeddings/index.md:components"

You can add them to your EDS-PDF pipeline by simply calling `add_pipe`, for instance:

<!-- no-check -->

```python
# ↑ Omitted code that defines the pipeline object ↑
pipeline.add_pipe("pdfminer-extractor", name="component-name", config=...)
```
