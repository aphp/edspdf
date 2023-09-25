# Data Structures


EDS-PDF stores PDFs and their annotation in a custom data structures that are
designed to be easy to use and manipulate. We must distinguish between:

- the data models used to store the PDFs and exchange them between the
  different components of EDS-PDF
- the tensors structures used to process the PDFs with deep learning models

## Itinerary of a PDF

A PDF is first converted to a [PDFDoc][edspdf.structures.PDFDoc] object, which contains the raw PDF content. This task is usually performed a [PDF extractor component](/components/extractors). Once the PDF is converted, the same object will be used and updated by the different components, and returned at the end of the pipeline.

When running a trainable component, the [PDFDoc][edspdf.structures.PDFDoc] is preprocessed and converted to tensors containing relevant features for the task. This task is performed in the `preprocess` method of the component. The resulting tensors are then collated together to form a batch, in the `collate` method of the component. After running the `forward` method of the component, the tensor predictions are finally assigned as annotations to original [PDFDoc][edspdf.structures.PDFDoc] objects in the `postprocess` method.


## Data models

The main data structure is the [PDFDoc][edspdf.structures.PDFDoc], which represents full a PDF document. It contains the raw PDF content, annotations for the full document, regardless of pages. A PDF is split into [Page][edspdf.structures.Page] objects that stores their number, dimension and optionally an image of the rendered page.

The PDF annotations are stored in [Box][edspdf.structures.Box] objects, which represent a rectangular region of the PDF. At the moment, box can only be specialized into [TextBox][edspdf.structures.TextBox] to represent text regions, such as lines extracted by a PDF extractor. Aggregated texts are stored in [Text][edspdf.structures.Text] objects, that are not associated with a specific box.

A [TextBox][edspdf.structures.TextBox] contains a list of [TextProperties][edspdf.structures.TextProperties] objects to store the style properties of a styled spans of the text.

??? note "Reference"

    ::: edspdf.structures
        options:
          heading_level: 3

## Tensor structure

The tensors used to process PDFs with deep learning models usually contain 4 main dimensions, in addition to the standard embedding dimensions:

- `samples`: one entry per PDF in the batch
- `pages`: one entry per page in a PDF
- `boxes`: one entry per box in a page
- `token`: one entry per token in a box (only for text boxes)

These tensors use a special [FoldedTensor](http://pypi.org/project/foldedtensor) format to store the data in a compact way and reshape the data depending on the requirements of a layer.
