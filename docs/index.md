# Overview

EDS-PDF provides modular framework to extract text information from PDF documents.

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

Let's build a simple PDF extractor that uses a rule-based classifier. There are two
ways to do this, either by using the [configuration system](#configuration) or by using
the pipeline API.

=== "Configuration based pipeline"

    Create a configuration file:

    ```toml title="config.cfg"
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

=== "API based pipeline"

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

See the [rule-based recipe](recipes/rule-based.md) for a step-by-step explanation of what is happening.

## Citation

If you use EDS-PDF, please cite us as below.

```bibtex
@article{gerardin_wajsburt_pdf,
  title={Bridging Clinical PDFs and Downstream Natural Language Processing: An Efficient Neural Approach to Layout Segmentation},
  author={G{\'e}rardin, Christel Ducroz and Wajsburt, Perceval and Dura, Basile and Calliger, Alice and Mouchet, Alexandre and Tannier, Xavier and Bey, Romain},
  journal={Available at SSRN 4587624}
}
```

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/) and
[AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
