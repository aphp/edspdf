# PopplerExtractor

We provide a PDF line extractor built on top of [PyMuPdf](https://pdfminersix.readthedocs.io/en/latest/).

The poppler software is more difficult to install than its `pdfminer` and `mupdf` counterparts.
In particular, the bindings we provide have not been tested on Windows.

!!! note "License"

    Beware, Poppler is distributed under the GPL license, therefore so is this
    component, and any model depending on this component must be too.

## Installation

For the licensing reason mentioned above, the `poppler` component is distributed
in a separate package `edspdf-poppler`. To install it, use your favorite Python package manager :

```bash
poetry add edspdf-poppler
# or
pip install edspdf-poppler
```

## Usage

```python
from edspdf import Pipeline
from pathlib import Path

# Add the component to a new pipeline
model = Pipeline()
model.add_pipe(
    "poppler-extractor",
    config=dict(
        extract_style=False,
    ),
)

# Apply on a new document
model(Path("path/to/your/pdf/document").read_bytes())
```

## Configuration

| Parameter       | Description                                                                           | Default |
|-----------------|---------------------------------------------------------------------------------------|---------|
| extract_style   | Whether to extract style (font, size, ...) information for each line of the document. | False   |
