![Tests](https://img.shields.io/github/actions/workflow/status/aphp/edspdf/tests.yml?branch=main&label=tests&style=flat-square)
[![Documentation](https://img.shields.io/github/actions/workflow/status/aphp/edspdf/documentation.yml?branch=main&label=docs&style=flat-square)](https://aphp.github.io/edspdf/latest/)
[![PyPI](https://img.shields.io/pypi/v/edspdf?color=blue&style=flat-square)](https://pypi.org/project/edspdf/)
[![Coverage](https://raw.githubusercontent.com/aphp/edspdf/coverage/coverage.svg)](https://raw.githubusercontent.com/aphp/edspdf/coverage/coverage.txt)
[![DOI](https://zenodo.org/badge/517726737.svg)](https://zenodo.org/badge/latestdoi/517726737)

# EDS-PDF

EDS-PDF provides a modular framework to extract text information from PDF documents.

You can use it out-of-the-box, or extend it to fit your specific use case. We provide a pipeline system and various utilities for visualizing and processing PDFs, as well as multiple components to build complex models:complex models:
- 📄 [Extractors](https://aphp.github.io/edspdf/latest/pipes/extractors) to parse PDFs (based on [pdfminer](https://github.com/euske/pdfminer), [mupdf](https://github.com/aphp/edspdf-mupdf) or [poppler](https://github.com/aphp/edspdf-poppler))
- 🎯 [Classifiers](https://aphp.github.io/edspdf/latest/pipes/box-classifiers) to perform text box classification, in order to segment PDFs
- 🧩 [Aggregators](https://aphp.github.io/edspdf/latest/pipes/aggregators) to produce an aggregated output from the detected text boxes
- 🧠 Trainable layers to incorporate machine learning in your pipeline (e.g., [embedding](https://aphp.github.io/edspdf/latest/pipes/embeddings) building blocks or a [trainable classifier](https://aphp.github.io/edspdf/latest/pipes/box-classifiers/trainable/))

Visit the [:book: documentation](https://aphp.github.io/edspdf/) for more information!

## Getting started

### Installation

Install the library with pip:

```bash
pip install edspdf
```

### Extracting text

Let's build a simple PDF extractor that uses a rule-based classifier. There are two
ways to do this, either by using the [configuration system](#configuration) or by using
the pipeline API.

Create a configuration file:

<h5 a><strong><code>config.cfg</code></strong></h5>

```ini
[pipeline]
pipeline = ["extractor", "classifier", "aggregator"]

[components.extractor]
@factory = "pdfminer-extractor"

[components.classifier]
@factory = "mask-classifier"
x0 = 0.2
x1 = 0.9
y0 = 0.3
y1 = 0.6
threshold = 0.1

[components.aggregator]
@factory = "simple-aggregator"
```

and load it from Python:

```python
import edspdf
from pathlib import Path

model = edspdf.load("config.cfg")  # (1)
```

Or create a pipeline directly from Python:

```python
from edspdf import Pipeline

model = Pipeline()
model.add_pipe("pdfminer-extractor")
model.add_pipe(
    "mask-classifier",
    config=dict(
        x0=0.2,
        x1=0.9,
        y0=0.3,
        y1=0.6,
        threshold=0.1,
    ),
)
model.add_pipe("simple-aggregator")
```

This pipeline can then be applied (for instance with this [PDF](https://github.com/aphp/edspdf/raw/main/tests/resources/letter.pdf)):

```python
# Get a PDF
pdf = Path("/Users/perceval/Development/edspdf/tests/resources/letter.pdf").read_bytes()
pdf = model(pdf)

body = pdf.aggregated_texts["body"]

text, style = body.text, body.properties
```

See the [rule-based recipe](https://aphp.github.io/edspdf/latest/recipes/rule-based) for a step-by-step explanation of what is happening.

## Citation

If you use EDS-PDF, please cite us as below.

```bibtex
@software{edspdf,
  author  = {Dura, Basile and Wajsburt, Perceval and Calliger, Alice and Gérardin, Christel and Bey, Romain},
  doi     = {10.5281/zenodo.6902977},
  license = {BSD-3-Clause},
  title   = {{EDS-PDF: Smart text extraction from PDF documents}},
  url     = {https://github.com/aphp/edspdf}
}
```

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/) and
[AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
