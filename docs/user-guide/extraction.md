# Extraction

The extraction phase consists of reading the PDF document and gather text blocs,
along with their dimensions and position within the document.
Said blocs will go on to the classification phase to separate the body from the rest.

| Method        | Description                              |
| ------------- | ---------------------------------------- |
| `pdfminer.v1` | Text-based PDF extraction using PDFMiner |

## Text-based PDF

We provide a unique extractor architecture, that is based on the
[PDFMiner](https://pdfminersix.readthedocs.io/en/latest/) library.
Be sure to have a look at their documentation, especially the
[part providing a bird's eye view of the PDF extraction process](https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html).

## Image-based PDF

Image-based PDF documents require an OCR[^1] step, which is not natively supported by EDS-PDF.
However, you can easily extend EDS-PDF by adding such a method to the registry, and using that
extractor function in place of `pdfminer.v1`. 

[^1]: Optical Character Recognition, or OCR, is the process of converting an image of text into a text format.
