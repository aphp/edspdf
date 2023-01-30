# MuPdfExtractor

We provide a PDF line extractor built on top of [PyMuPdf](https://pdfminersix.readthedocs.io/en/latest/).

This extractor is the fastest but may not be as portable as [PdfMiner](./pdfminer).
However, it should also be relatively easy to install on a wide range of architectures, Linux, OS X and Windows.

!!! note "License"

    Beware, PyMuPdf is distributed under the AGPL license, therefore so is this
    component, and any model depending on this component must be too.

## Installation

For the licensing reason mentioned above, the `mupdf` component is distributed
in a separate package `edspdf-mupdf`. To install it, use your favorite Python package manager :

```bash
poetry add edspdf-mupdf
# or
pip install edspdf-mupdf
```

## Usage

```python
from edspdf import Pipeline
from pathlib import Path

# Add the component to a new pipeline
model = Pipeline()
model.add_pipe(
    "mupdf-extractor",
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
