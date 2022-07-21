# Trained Pipeline with Scikit-Learn

In this section, we'll see how we can train a machine-learning based
classifier to get better performances. In this example, we will use
a Scikit-Learn _pipeline_.

!!! warning

    Scikit-Learn is ill-equipped to deal with text data. As such,
    it is not the best candidate to provide an effective classification
    method. However, it can still perform quite well and remains a good place
    to start tinkering with the inner workings of EDS-PDF.

## PDF annotation

See the [PDF annotation recipe](annotation.md) for one annotation methodology.
For the rest of this recipe, we will consider that the dataset follows the same structure.

## Pipeline definition

Let's use the following pipeline:

```toml title="config.cfg"
[reader]
@readers = "pdf-reader.v1"

[reader.extractor]
@extractors = "pdfminer-extractor.v1"

[reader.transform]
@transforms = "chain.v1"

[reader.transform.*.dates]
@transforms = "dates.v1"

[reader.transform.*.telephone]
@transforms = "telephone.v1"

[reader.transform.*.dimensions]
@transforms = "dimensions.v1"

# The model has not been trained yet
# We still reference it to make sure we use the same configuration
[reader.classifier]
@classifiers = "sklearn.v1"
path = "classifier.joblib"

[reader.aggregator]
@aggregators = "styled.v1"
```

## Data preparation

The reader object exposes a [`#!python prepare_data`][edspdf.reading.reader.PdfReader.prepare_data] method,
which runs the pipeline until the classification phase, and returns the `DataFrame` as it would be seen
by the classifier. Hence, we can use it to produce a training dataset for the classification step.

It means that we can use the same configuration for preparing the training data for the classifier and for the full pipeline,
guaranteeing that the data will be correctly pre-processed at runtime.

```python
# ↑ Omitted code from the annotation recipe ↑

import json
import pandas as pd

from edspdf import registry, Config
from edspdf.reading import PdfReader
from edspdf.classification.align import align_labels

from pathlib import Path


def prepare_dataset(
    reader: PdfReader,
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

    for path in directory.glob("*.pdf"):
        meta = json.loads(path.with_suffix(".json").read_text())
        del meta["annotations"]

        df = reader(path.read_bytes(), **meta)

        dfs.append(df)

    return pd.concat(dfs)


config = Config().from_disk("config.cfg")
del config["reader"]["classifier"]  # (1)

reader = registry.resolve(config)["reader"]

path = Path("dataset/train")

annotations = get_annotations(path)  # (2)
lines = prepare_dataset(reader, path)

annotated = align_labels(lines=lines, labels=annotations, threshold=0.8)  # (3)

annotated.to_csv("data.csv", index=False)
```

1. We remove the classifier from the pipeline definition since it is not defined at this point.
2. See the [PDF annotation recipe](annotation.md)
3. The object `annotated` now contains every text bloc that was covered by an annotated region,
   along with its label.

## Training the machine learning pipeline

Now everything is ready to train a Scikit-Learn pipeline! Let's define a simple classifier:

```python title="classifier.py"
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

n_components = 20
max_features = 2000
seed = 0


text_vectorizer = Pipeline(
    [
        ("vect", CountVectorizer(strip_accents="ascii", max_features=max_features)),
        ("tfidf", TfidfTransformer()),
        ("reduction", TruncatedSVD(n_components=n_components, random_state=seed)),
    ]
)

classifier = Pipeline(
    [
        ("norm", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=seed)),
    ]
)

pipeline = Pipeline(
    [
        (
            "union",
            ColumnTransformer(
                [
                    ("text", text_vectorizer, "text"),
                    (
                        "others",
                        "passthrough",
                        [
                            "page",
                            "x0",
                            "x1",
                            "y0",
                            "y1",
                            "telephone",
                            "date",
                            "width",
                            "height",
                            "area",
                        ],
                    ),
                ]
            ),
        ),
        ("classifier", classifier),
    ]
)
```

And train it:

```python
import pandas as pd

from joblib import dump
from classifier import pipeline


data = pd.read_csv("data.csv")
X_train, Y_train = data.drop(columns=["label"]), data["label"]

pipeline.fit(X_train, Y_train)

dump(pipeline, "classifier.joblib")
```

## Using the full pipeline

Now that the machine learning model is trained, we can use the full pipeline:

```python
import edspdf
from pathlib import Path

reader = edspdf.load("config.cfg")

# Get a PDF
pdf = Path("letter.pdf").read_bytes()

texts = reader(pdf)

texts["body"]
# Out: Cher Pr ABC, Cher DEF,\n...
```
