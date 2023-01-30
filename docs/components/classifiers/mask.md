# Mask Classification

We developed a simple classifier that roughly uses the same strategy as PDFBox, namely:

- define a "mask" on the PDF documents ;
- keep every text bloc within that mask, tag everything else as pollution.

## Factories

Two factories are available in the `classifiers` registry: `mask-classifier` and `multi-mask-classifier`.

### `mask-classifier`

The simplest form. You define the mask, everything else is tagged as pollution.

Example configuration :

=== "API-based"

    ```python
    pipeline.add_pipe(
        "mask-classifier",
        name="classifier",
        config={
            "threshold": 0.9,
            "x0": 0.1,
            "y0": 0.1,
            "x1": 0.9,
            "y1": 0.9,
        },
    )
    ```

=== "Configuration-based"

    ```ini
    [components.classifier]
    @classifiers = "mask-classifier"
    x0 = 0.1
    y0 = 0.1
    x1 = 0.9
    y1 = 0.9
    threshold = 0.9
    ```

### `multi-mask-classifier`

A generalisation, wherein the user defines a number of regions.

The following configuration produces _exactly_ the same classifier as `mask.v1` example above.

=== "API-based"

    ```python
    pipeline.add_pipe(
        "multi-mask-classifier",
        name="classifier",
        config={
            "threshold": 0.9,
            "body": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.3, "label": "header"},
        },
    )
    ```

=== "Configuration-based"

    ```ini
    [components.classifier]
    @factory = "multi-mask-classifier"
    threshold = 0.9

    [components.classifier.body]
    label = "body"
    x0 = 0.1
    y0 = 0.1
    x1 = 0.9
    y1 = 0.9
    ```

The following configuration defines a `header` region.

=== "API-based"

    ```python
    pipeline.add_pipe(
        "multi-mask-classifier",
        name="classifier",
        config={
            "threshold": 0.9,
            "body": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.3, "label": "header"},
            "header": {"x0": 0.1, "y0": 0.3, "x1": 0.9, "y1": 0.9, "label": "body"},
        },
    )
    ```

=== "Configuration-based"

    ```ini
    [components.classifier]
    @factory = "multi-mask-classifier"
    threshold = 0.9

    [components.classifier.header]
    label = "header"
    x0 = 0.1
    y0 = 0.1
    x1 = 0.9
    y1 = 0.3

    [components.classifier.body]
    label = "body"
    x0 = 0.1
    y0 = 0.3
    x1 = 0.9
    y1 = 0.9
    ```

Any bloc that is not part of a mask is tagged as `pollution`.
