# Extraction

The extraction phase consists of reading the PDF document and gather text blocs, along with their dimensions and position within the document. Said blocs will go on to the classification phase to separate the body from the rest.

## Text-based PDF

We provide a multiple extractor architectures for text-based PDFs :

<!-- --8<-- [start:components] -->

| Factory name                                                               | Description                                     |
|----------------------------------------------------------------------------|-------------------------------------------------|
| [`pdfminer-extractor`][edspdf.pipes.extractors.pdfminer.PdfMinerExtractor] | Extracts text lines with the `pdfminer` library |
| [`mupdf-extractor`][edspdf_mupdf.MuPdfExtractor]                           | Extracts text lines with the `pymupdf` library  |
| [`poppler-extractor`][edspdf_poppler.PopplerExtractor]                     | Extracts text lines with the `poppler` library  |

<!-- --8<-- [end:components] -->

## Image-based PDF

Image-based PDF documents require an OCR[^1] step, which is not natively supported by EDS-PDF.
However, you can easily extend EDS-PDF by adding such a method to the registry.

We plan on adding such an OCR extractor component in the future.

[^1]: Optical Character Recognition, or OCR, is the process of extracting characters and words from an image.
