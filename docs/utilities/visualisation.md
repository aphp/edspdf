# Visualisation

EDS-PDF provides utilities to help you visualise the output of the pipeline.

## Visualising a pipeline's output

You can use EDS-PDF to overlay labelled bounding boxes on top of a PDF document.

```python
import edspdf
from pathlib import Path
from edspdf.visualization.annotations import show_annotations

config = """
[reader]
@readers = "pdf-reader.v1"

[reader.extractor]
@extractors = "pdfminer.v1"

[reader.classifier]
@classifiers = "mask.v1"
x0 = 0.1
x1 = 0.9
y0 = 0.4
y1 = 0.9
threshold = 0.1

[reader.aggregator]
@aggregators = "simple.v1"
"""

reader = edspdf.from_str(config)

# Get a PDF
pdf = Path("letter.pdf").read_bytes()

# Construct the DataFrame of blocs
lines = reader.prepare_and_predict(pdf)

# Compute an image representation of each page of the PDF
# overlaid with the predicted bounding boxes
imgs = show_annotations(pdf=pdf, annotations=lines)

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
from edspdf.visualization.merge import merge_lines

merged = merge_lines(lines)

imgs = show_annotations(pdf=pdf, annotations=merged)
imgs[0]
```

See the difference:

=== "Original"

    ![lines](resources/lines.jpeg)

=== "Merged"

    ![lines](resources/merged.jpeg)

The `merge_lines` method uses the notion of maximal cliques to compute merges.
It forbids the combined blocs from overlapping with any bloc from another label.
