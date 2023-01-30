# PDF Annotation

In this section, we will cover one methodology to annotate PDF documents.

!!! aphp "Data annotation at AP-HP's CDW"

    At AP-HP's CDW[^1], we recently moved away from a rule- and Java-based PDF extraction pipeline
    (using PDFBox) to one using EDS-PDF. Hence, EDS-PDF is used in production, helping
    extract text from around 100k PDF documents every day.

    To train our pipeline presently in production, we annotated **around 270 documents**, and reached
    a **f1-score of 0.98** on the body classification.

## Preparing the data for annotation

We will frame the annotation phase as an image segmentation task,
where annotators are asked to draw bounding boxes around the different sections.
Hence, the very first step is to convert PDF documents to images. We suggest using the
library `pdf2image` for that step.

The following script will convert the PDF documents located in a `data/pdfs` directory
to PNG images inside the `data/images` folder.

```python
import pdf2image
from pathlib import Path

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"

for pdf in PDF_DIR.glob("*.pdf"):
    imgs = pdf2image.convert_from_bytes(pdf)

    for page, img in enumerate(imgs):
        path = IMAGE_DIR / f"{pdf.stem}_{page}.png"
        img.save(path)
```

You can use any annotation tool to annotate the images. If you're looking for a simple
way to annotate from within a Jupyter Notebook,
[ipyannotations](https://ipyannotations.readthedocs.io/en/latest/examples/image-landmarks.html#annotating-bounding-boxes)
might be a good fit.

You will need to post-process the output
to convert the annotations to the following format:

| Key     | Description                                                        |
|---------|--------------------------------------------------------------------|
| `page`  | Page within the PDF (0-indexed)                                    |
| `x0`    | Horizontal position of the top-left corner of the bounding box     |
| `x1`    | Horizontal position of the bottom-right corner of the bounding box |
| `y0`    | Vertical position of the top-left corner of the bounding box       |
| `y1`    | Vertical position of the bottom-right corner of the bounding box   |
| `label` | Class of the bounding box (eg `body`, `header`...)                 |

All dimensions should be normalised by the height and width of the page.

## Saving the dataset

Once the annotation phase is complete, make sure the train/test split is performed
once and for all when you create the dataset.

We suggest the following structure:

```title="Directory structure"
dataset/
├── train/
│   ├── <note_id_1>.pdf
│   ├── <note_id_1>.json
│   ├── <note_id_2>.pdf
│   ├── <note_id_2>.json
│   └── ...
└── test/
    ├── <note_id_n>.pdf
    ├── <note_id_n>.json
    └── ...
```

Where the normalised annotation resides in a JSON file living next to the related PDF,
and uses the following schema:

| Key            | Description                                     |
| -------------- | ----------------------------------------------- |
| `note_id`      | Reference to the document                       |
| `<properties>` | Optional property of the document itself        |
| `annotations`  | List of annotations, following the schema above |

This structure presents the advantage of being machine- and human-friendly.
The JSON file contains annotated regions as well as any document property that
could be useful to adapt the pipeline (typically for the classification step).

## Extracting annotations

The following snippet extracts the annotations into a workable format:

```python
from pathlib import Path
import pandas as pd


def get_annotations(
    directory: Path,
) -> pd.DataFrame:
    """
    Read annotations from the dataset directory.

    Parameters
    ----------
    directory : Path
        Dataset directory

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the annotations.
    """
    dfs = []

    iterator = tqdm(list(directory.glob("*.json")))

    for path in iterator:
        meta = json.loads(path.read_text())
        df = pd.DataFrame.from_records(meta.pop("annotations"))

        for k, v in meta.items():  # (1)
            df[k] = v

        dfs.append(df)

    return pd.concat(dfs)


train_path = Path("dataset/train")

annotations = get_annotations(train_path)
```

1. Add a column for each additional property saved in the dataset.

The annotations compiled this way can be used to train a pipeline.
See the [trained pipeline recipe](./training.md) for more detail.

[^1]: Greater Paris University Hospital's Clinical Data Warehouse
