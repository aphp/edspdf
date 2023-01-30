# Alternatives & Comparison

EDS-PDF was developed to propose a more modular and extendable approach to PDF extraction than [PDFBox](https://pdfbox.apache.org/), the legacy implementation at APHP's clinical data warehouse.

EDS-PDF takes inspiration from Explosion's [spaCy](https://spacy.io) pipelining system and closely follows its API. Therefore, the core object within EDS-PDF is the Pipeline, which organises the processing of PDF documents into multiple components. However, unlike spaCy, the library is built around a single deep learning framework, pytorch, which makes model development easier. Similar to [spaCy](https://spacy.io) and [thinc](https://thinc.ai), EDS-PDF also relies on [catalogue](https://github.com/explosion/catalogue) entry points and a custom powerful custom configuration system. This allows complex pipeline models to be built and trained using either the library API or configuration files.
