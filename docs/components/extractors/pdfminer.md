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
    "pdfminer-extractor",
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
| line_overlap    | See PDFMiner documentation                                                            | 0.5     |
| char_margin     | See PDFMiner documentation                                                            | 2.05    |
| line_margin     | See PDFMiner documentation                                                            | 0.5     |
| word_margin     | See PDFMiner documentation                                                            | 0.1     |
| boxes_flow      | See PDFMiner documentation                                                            | 0.5     |
| detect_vertical | See PDFMiner documentation                                                            | False   |
| all_texts       | See PDFMiner documentation                                                            | False   |
| extract_style   | Whether to extract style (font, size, ...) information for each line of the document. | False   |
