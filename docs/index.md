# Overview

EDS-PDF provides modular framework to extract text from PDF documents.

You can use it out-of-the-box, or extend it to fit your use-case.

## Getting started

### Installation

Install the library with pip:

<div class="termy">

```console
$ pip install edspdf
---> 100%
color:green Installation successful
```

</div>

### Extracting text

Let's build a simple PDF extractor that uses a rule-based classifier,
using the following configuration:

```toml title="config.cfg"
[reader]
@readers = "pdf-reader.v1"

[reader.extractor]
@extractors = "pdfminer.v1"

[reader.classifier]
@classifiers = "mask.v1"
x0 = 0.2
x1 = 0.9
y0 = 0.3
y1 = 0.6
threshold = 0.1

[reader.aggregator]
@aggregators = "simple.v1"
```

The PDF reader can be instantiated and applied (for instance with this [PDF](https://github.com/aphp/edspdf/raw/master/tests/resources/letter.pdf)):

```python
import edspdf
from pathlib import Path

reader = edspdf.load("config.cfg")  # (1)

# Get a PDF
pdf = Path("letter.pdf").read_bytes()

texts = reader(pdf)

texts["body"]
# Out: Cher Pr ABC, Cher DEF,\n...
```

1. The `reader` object is loaded from the configuration directly.

See the [rule-based recipe](recipes/rules.md) for a step-by-step explanation of what is happening.

## Citation

If you use EDS-NLP, please cite us as below.

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
