# Visualisation

EDS-PDF provides utilities to help you visualise the output of the pipeline.

## Visualising a pipeline's output

You can use EDS-PDF to overlay labelled bounding boxes on top of a PDF document.

```python
import edspdf
from confit import Config
from pathlib import Path
from edspdf.visualization import show_annotations

config = """
[pipeline]
pipeline = ["extractor", "classifier"]

[components]

[components.extractor]
@factory = "pdfminer-extractor"
extract_style = true

[components.classifier]
@factory = "mask-classifier"
x0 = 0.25
x1 = 0.95
y0 = 0.3
y1 = 0.9
threshold = 0.1
"""

model = edspdf.load(Config.from_str(config))

# Get a PDF
pdf = Path("/Users/perceval/Development/edspdf/tests/resources/letter.pdf").read_bytes()

# Construct the DataFrame of blocs
doc = model(pdf)

# Compute an image representation of each page of the PDF
# overlaid with the predicted bounding boxes
imgs = show_annotations(pdf=pdf, annotations=doc.text_boxes)

imgs[0]
```

If you run this code in a Jupyter notebook, you'll see the following:

![lines](resources/lines.jpeg)

## Merging blocs together

To help debug a pipeline (or a labelled dataset), you might want to
merge blocs together according to their labels. EDS-PDF provides a `merge_lines` method
that does just that.

```python
# ↑ Omitted code above ↑
from edspdf.visualization import merge_boxes, show_annotations

merged = merge_boxes(doc.text_boxes)

imgs = show_annotations(pdf=pdf, annotations=merged)
imgs[0]
```

See the difference:

=== "Original"

    ![lines](resources/lines.jpeg)

=== "Merged"

    ![lines](resources/merged.jpeg)

The `merge_boxes` method uses the notion of maximal cliques to compute merges.
It forbids the combined blocs from overlapping with any bloc from another label.
