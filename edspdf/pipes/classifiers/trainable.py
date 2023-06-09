import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Set

import torch
import torch.nn.functional as F
from foldedtensor import as_folded_tensor
from sklearn.metrics import classification_report
from tqdm import tqdm

from edspdf.layers.vocabulary import Vocabulary
from edspdf.pipeline import Pipeline
from edspdf.pipes.embeddings import EmbeddingOutput
from edspdf.registry import registry
from edspdf.structures import PDFDoc
from edspdf.trainable_pipe import Scorer, TrainablePipe
from edspdf.utils.torch import ActivationFunction, get_activation_function


def classifier_scorer(pairs):
    return classification_report(
        [b.label for pred, gold in pairs for b in gold.text_boxes if b.text != ""],
        [b.label for pred, gold in pairs for b in pred.text_boxes if b.text != ""],
        output_dict=True,
        zero_division=0,
    )


@registry.factory.register("trainable-classifier")
class TrainableClassifier(TrainablePipe[Dict[str, Any]]):
    """
    This component predicts a label for each box over the whole document using machine
    learning.

    !!! note

        You must train the model your model to use this classifier.
        See [Model training][training-a-pipeline] for more information

    Examples
    --------

    The classifier is composed of the following blocks:

    - a configurable box embedding layer
    - a linear classification layer

    In this example, we use a `box-embedding` layer to generate the embeddings
    of the boxes. It is composed of a text encoder that embeds the text features of the
    boxes and a layout encoder that embeds the layout features of the boxes.
    These two embeddings are summed and passed through an optional `contextualizer`,
    here a `box-transformer`.

    === "API-based"

        ```python
        pipeline.add_pipe(
            "trainable-classifier",
            name="classifier",
            config={
                # simple embedding computed by pooling embeddings of words in each box
                "embedding": {
                    "@factory": "sub-box-cnn-pooler",
                    "out_channels": 64,
                    "kernel_sizes": (3, 4, 5),
                    "embedding": {
                        "@factory": "simple-text-embedding",
                        "size": 72,
                    },
                },
                "labels": ["body", "pollution"],
                "activation": "relu",
            },
        )
        ```

    === "Configuration-based"

        ```toml
        [components.classifier]
        @factory = "trainable-classifier"
        labels = ["body", "pollution"]
        activation = "relu"

        [components.classifier.embedding]
        @factory = "sub-box-cnn-pooler"
        out_channels = 64
        kernel_sizes = (3, 4, 5)

        [components.classifier.embedding.embedding]
        @factory = "simple-text-embedding"
        size = 72
        ```

    Parameters
    ----------
    labels: Sequence[str]
        Initial labels of the classifier (will be completed during initialization)
    embedding: TrainablePipe[EmbeddingOutput]
        Embedding module to encode the PDF boxes
    activation: ActivationFunction
        Name of the activation function
    dropout_p: float
        Dropout probability used on the output of the box and textual encoders
    scorer: Scorer
        Scoring function
    """

    def __init__(
        self,
        embedding: TrainablePipe[EmbeddingOutput],
        labels: Sequence[str] = ("pollution",),
        activation: ActivationFunction = "gelu",
        dropout_p: float = 0.0,
        scorer: Scorer = classifier_scorer,
        pipeline: Pipeline = None,
        name: str = "trainable-classifier",
    ):
        super().__init__(pipeline, name)
        self.label_voc: Vocabulary = Vocabulary(list(dict.fromkeys(labels)))
        self.embedding = embedding

        size = self.embedding.output_size

        self.linear = torch.nn.Linear(size, size)
        self.classifier = torch.nn.Linear(
            in_features=self.embedding.output_size,
            out_features=len(self.label_voc),
        )
        self.activation = get_activation_function(activation)
        self.dropout = torch.nn.Dropout(dropout_p)

        # Scoring function
        self.score = scorer

    def post_init(self, gold_data: Iterable[PDFDoc], exclude: set):
        if self.name in exclude:
            return
        exclude.add(self.name)
        self.embedding.post_init(gold_data, exclude)

        label_voc_indices = dict(self.label_voc.indices)

        with self.label_voc.initialization():
            for doc in tqdm(gold_data, desc="Initializing classifier"):
                self.preprocess_supervised(doc)

        self.update_weights_from_vocab_(label_voc_indices)

    def update_weights_from_vocab_(self, label_voc_indices):
        label_indices = dict(
            (
                *label_voc_indices.items(),
                *self.label_voc.indices.items(),
            )
        )
        old_index = [label_voc_indices[label] for label in label_voc_indices]
        new_index = [label_indices[label] for label in label_voc_indices]
        new_linear = torch.nn.Linear(
            self.embedding.output_size,
            len(label_indices),
        )
        new_linear.weight.data[new_index] = self.linear.weight.data[old_index]
        new_linear.bias.data[new_index] = self.linear.bias.data[old_index]
        self.classifier = new_linear

    def preprocess(self, doc: PDFDoc) -> Dict[str, Any]:
        result = {
            "embedding": self.embedding.preprocess(doc),
            "doc_id": doc.id,
        }
        return result

    def preprocess_supervised(self, doc: PDFDoc) -> Dict[str, Any]:
        return {
            **self.preprocess(doc),
            "labels": [
                [
                    self.label_voc.encode(b.label) if b.label is not None else -100
                    for b in page.text_boxes
                ]
                for page in doc.pages
            ],
        }

    def collate(self, batch, device: torch.device) -> Dict:
        collated = {
            "embedding": self.embedding.collate(batch["embedding"], device),
            "doc_id": batch["doc_id"],
        }
        if "labels" in batch:
            collated.update(
                {
                    "labels": as_folded_tensor(
                        batch["labels"],
                        data_dims=("line",),
                        full_names=("sample", "page", "line"),
                        device=device,
                        dtype=torch.long,
                    ),
                }
            )

        return collated

    def forward(self, batch: Dict) -> Dict:
        embedding_res = self.embedding.module_forward(batch["embedding"])
        embeddings = embedding_res["embeddings"]

        output = {"loss": 0, "mask": embeddings.mask}

        # Label prediction / learning
        logits = self.classifier(embeddings).refold("line")
        if "labels" in batch:
            targets = batch["labels"].refold(logits.data_dims)
            output["label_loss"] = (
                F.cross_entropy(
                    logits,
                    targets,
                    reduction="sum",
                )
                / targets.mask.sum()
            )
            output["loss"] = output["loss"] + output["label_loss"]
        else:
            output["logits"] = logits
            output["labels"] = logits.argmax(-1)

        return output

    def postprocess(self, docs: Sequence[PDFDoc], output: Dict) -> Sequence[PDFDoc]:
        for b, label in zip(
            (b for doc in docs for b in doc.text_boxes),
            output["labels"].tolist(),
        ):
            b.label = self.label_voc.decode(label) if b.text != "" else None
        return docs

    def save_extra_data(self, path: Path, exclude: Set):
        if self.name in exclude:
            return
        exclude.add(self.name)

        self.embedding.save_extra_data(path / "embedding", exclude)

        os.makedirs(path, exist_ok=True)

        with (path / "label_voc.json").open("w") as f:
            json.dump(self.label_voc.indices, f)

    def load_extra_data(self, path: Path, exclude: Set):
        if self.name in exclude:
            return
        exclude.add(self.name)

        self.embedding.load_extra_data(path / "embedding", exclude)

        label_voc_indices = dict(self.label_voc.indices)

        with (path / "label_voc.json").open("r") as f:
            self.label_voc.indices = json.load(f)

        self.update_weights_from_vocab_(label_voc_indices)
