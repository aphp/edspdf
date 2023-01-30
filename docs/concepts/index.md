# Concepts

The goal of EDS-PDF is to provide a **framework** for processing PDF documents, along with some utilities and a few components, stitched together by a robust pipeline and configuration system.

Processing PDFs usually involves many steps such as extracting lines, running OCR models, detecting and classifying boxes, filtering and aggregating parts of the extracted texts, etc. Organising these steps together, combining static and deep learning components, while remaining modular and efficient is a challenge. This is why EDS-PDF is built on top of a new pipelining system.

At the moment, three types of components are implemented in the library:

1. **extraction** components extract lines from a raw PDF and return a `PDFDoc` object filled with these text boxes.
2. **classification** components classify each box with labels, such as `body`, `header`, `footer`...
3. **aggregation** components compiles the lines together according to their classes to re-create the original text.

EDS-PDF takes inspiration from Explosion's [spaCy](https://spacy.io) pipelining system and closely follows its API. Therefore, the core object within EDS-PDF is the Pipeline, which organises the processing of PDF documents into multiple components. However, unlike spaCy, the library is built around a single deep learning framework, pytorch, which makes model development easier.

Similar to [spaCy](https://spacy.io) and [thinc](https://thinc.ai), EDS-PDF also relies on [catalogue](https://github.com/explosion/catalogue) entry points and a custom powerful custom configuration system. This allows complex pipeline models to be built and trained using either the library API or configuration files.

In the next chapters, we will go over some core concepts of EDS-PDF:

- the [pipelining system](./pipeline.md)
- the [configuration system](./configuration.md)
- deep learning support with [trainable components](./trainable-components.md)
