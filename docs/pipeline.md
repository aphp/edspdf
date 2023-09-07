# Pipeline {: #edspdf.pipeline.Pipeline }

The goal of EDS-PDF is to provide a **framework** for processing PDF documents, along with some utilities and a few components, stitched together by a robust pipeline and configuration system.

Processing PDFs usually involves many steps such as extracting lines, running OCR models, detecting and classifying boxes, filtering and aggregating parts of the extracted texts, etc. Organising these steps together, combining static and deep learning components, while remaining modular and efficient is a challenge. This is why EDS-PDF is built on top of a new pipelining system.


!!! note "Deep learning frameworks"

    The EDS-PDF trainable components are built around the PyTorch framework. While you
    can use any technology in static components, we do not provide tools to train
    components built with other deep learning frameworks.

## Creating a pipeline

A pipe is a processing block (like a function) that applies a transformation on its input and returns a modified object.

At the moment, four types of pipes are implemented in the library:

1. **extraction** components extract lines from a raw PDF and return a [`PDFDoc`][edspdf.structures.PDFDoc] object filled with these text boxes.
2. **classification** components classify each box with labels, such as `body`, `header`, `footer`...
3. **aggregation** components compiles the lines together according to their classes to re-create the original text.
4. **embedding** components don't directly update the annotations on the document but have specific deep-learning methods (see the [TrainablePipe][edspdf.trainable_pipe.TrainablePipe] page) that can be composed to form a machine learning model.

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

For more information on how to use the pipeline, refer to the [Inference](/inference) page.

### Hybrid models

EDS-PDF was designed to facilitate the training and inference of hybrid models that
arbitrarily chain static components or trained deep learning components. Static components are callable objects that take a PDFDoc object as input, perform arbitrary transformations over the input, and return the modified object. [Trainable pipes][edspdf.trainable_pipe.TrainablePipe], on the other hand, allow for deep learning operations to be performed on the [PDFDoc][edspdf.structures.PDFDoc] object and must be trained to be used.

## Saving and loading a pipeline

Pipelines can be saved and loaded using the `save` and `load` methods. The saved pipeline is not a pickled objet but a folder containing the config file, the weights and extra resources for each pipeline. This allows for easy inspection and modification of the pipeline, and avoids the execution of arbitrary code when loading a pipeline.

```python
model.save("path/to/your/model")
model = edspdf.load("path/to/your/model")
```

To share the pipeline and turn it into a pip installable package, you can use the `package` method, which will use or create a pyproject.toml file, fill it accordingly, and create a wheel file. At the moment, we only support the poetry package manager.

```python
model.package(
    name="your-package-name",  # leave None to reuse name in pyproject.toml
    version="0.0.1",
    root_dir="path/to/project/root",  # optional, to retrieve an existing pyproject.toml file
    # if you don't have a pyproject.toml, you can provide the metadata here instead
    metadata=dict(
        authors="Firstname Lastname <your.email@domain.fr>",
        description="A short description of your package",
    ),
)
```

This will create a wheel file in the root_dir/dist folder, which you can share and install with pip
