# Extraction

The extraction phase consists of reading the PDF document and gather text blocs,
along with their dimensions and position within the document.
Said blocs will go on to the classification phase to separate the body from the rest.

## Text-based PDF

We provide a multiple extractor architectures for text-based PDFs :

| Component                        | Description                              |
|----------------------------------|------------------------------------------|
| [pdfminer-extractor](./pdfminer) | Text-based PDF extraction using PDFMiner |
| [mupdf-extractor](./mupdf)       | Text-based PDF extraction using MuPDF    |
| [poppler-extractor](./poppler)   | Text-based PDF extraction using Poppler  |

## Image-based PDF

Image-based PDF documents require an OCR[^1] step, which is not natively supported by EDS-PDF.
However, you can easily extend EDS-PDF by adding such a method to the registry.

We plan on adding such an OCR extractor component in the future.

[^1]: Optical Character Recognition, or OCR, is the process of extracting characters and words from an image.
