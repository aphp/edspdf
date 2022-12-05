# PdfMinerExtractor

We provide a PDF line extractor built on top of [PdfMiner](https://pdfminersix.readthedocs.io/en/latest/).

This is the most portable extractor, since it is pure-python and can therefore be run on any platform.
Be sure to have a look at their documentation, especially
the [part providing a bird's eye view of the PDF extraction process](https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html).

## Usage

```python
from edspdf import Pipeline
from pathlib import Path

# Add the component to a new pipeline
model = Pipeline()
model.add_pipe(
    "pdfminer",
    config=dict(
        extract_styles=False,
    ),
)

# Apply on a new document
model(Path("path/to/your/pdf/document").read_bytes())
```

## Configuration

| Parameter       | Description                                                                           | Default |
|-----------------|---------------------------------------------------------------------------------------|---------|
| line_overlap    | See PDFMiner documentation                                                            | 0.5     |
| char_margin     | See PDFMiner documentation                                                            | 2.05    |
| line_margin     | See PDFMiner documentation                                                            | 0.5     |
| word_margin     | See PDFMiner documentation                                                            | 0.1     |
| boxes_flow      | See PDFMiner documentation                                                            | 0.5     |
| detect_vertical | See PDFMiner documentation                                                            | False   |
| all_texts       | See PDFMiner documentation                                                            | False   |
| extract_style   | Whether to extract style (font, size, ...) information for each line of the document. | False   |

The `mupdf` extractor, based on the [python version MuPDF](https://github.com/pymupdf/PyMuPDF) is
faster and should also be relatively easy to install on a wide range of architectures, Linux, OS X and Windows. Indeed, the package offering a large number of pre-compiled binaries
on [PyPI](https://pypi.org/project/PyMuPDF/#files).

Finally, the `poppler` extractor based on Poppler is the fastest extraction system but is more difficult
to install. In particular, the bindings we provide have not been tested on Windows.
