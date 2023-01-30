# Pipeline

The core object within EDS-PDF is the Pipeline, which organises the processing of PDF documents into multiple components.

!!! note "Deep learning frameworks"

    The EDS-PDF trainable components are built around the PyTorch framework. While you
    can use any technology in static components, we do not provide tools to train
    components built with other deep learning frameworks.

A component is a processing block (like a function) that applies a transformation on its input and returns a modified object.

At the moment, three types of components are implemented in the library:

1. **extraction** components extract lines from a raw PDF and return a `PDFDoc` object filled with these text boxes.
2. **classification** components classify each box with labels, such as `body`, `header`, `footer`...
3. **aggregation** components compiles the lines together according to their classes to re-create the original text.

To create your first pipeline, execute the following code:

```python
from edspdf import Pipeline

model = Pipeline()
# will extract text lines from a document
model.add_pipe(
    "pdfminer-extractor",
    config=dict(
        extract_style=False,
    ),
)
# classify everything inside the `body` bounding box as `body`
model.add_pipe(
    "mask-classifier", config=dict(body={"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9})
)
# aggregates the lines together to re-create the original text
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

## Hybrid models

EDS-PDF was designed to facilitate the training and inference of hybrid models that
arbitrarily chain static components or trained deep learning components. Static components are callable objects that take a PDFDoc object as input, perform arbitrary transformations over the input, and return the modified object. [Trainable components](./trainable-components.md), on the other hand, allow for deep learning operations to be performed on the PDFDoc object and must be trained to be used.
