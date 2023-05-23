import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Set

import torch
import torch.nn.functional as F
from foldedtensor import as_folded_tensor
from sklearn.metrics import classification_report
from tqdm import tqdm

from edspdf.component import Scorer, TorchComponent
from edspdf.components.embeddings import EmbeddingOutput
from edspdf.layers.vocabulary import Vocabulary
from edspdf.pipeline import Pipeline
from edspdf.registry import registry
from edspdf.structures import PDFDoc
from edspdf.utils.torch import ActivationFunction, get_activation_function


def classifier_scorer(pairs):
    return classification_report(
        [b.label for pred, gold in pairs for b in gold.text_boxes if b.text != ""],
        [b.label for pred, gold in pairs for b in pred.text_boxes if b.text != ""],
        output_dict=True,
        zero_division=0,
    )


@registry.factory.register("trainable_classifier")
class TrainableClassifier(TorchComponent[Dict[str, Any]]):
    """
    Runs a deep learning classifier model on the boxes.

    Parameters
    ----------
    labels: Sequence[str]
        Initial labels of the classifier (will be completed during initialization)
    embedding: TorchComponent[EmbeddingOutput]
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
        embedding: TorchComponent[EmbeddingOutput],
        labels: Sequence[str] = ("pollution",),
        activation: ActivationFunction = "gelu",
        dropout_p: float = 0.15,
        scorer: Scorer = classifier_scorer,
        pipeline: Pipeline = None,
        name: str = "trainable_classifier",
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
        logits = self.classifier(embeddings)
        if "labels" in batch:
            targets = batch["labels"]
            output["label_loss"] = F.cross_entropy(
                logits.view(targets.numel(), -1),
                targets,
                reduction="sum",
            )
            output["loss"] = output["loss"] + output["label_loss"]
        else:
            output["logits"] = logits
            output["labels"] = logits.argmax(-1)

        return output

    def postprocess(self, docs: Sequence[PDFDoc], output: Dict) -> Sequence[PDFDoc]:
        for b, label in zip(
            (b for doc in docs for b in doc.text_boxes),
            output["labels"][output["mask"]].tolist(),
        ):
            b.label = self.label_voc.decode(label) if b.text != "" else None
        return docs

    def save_extra_data(self, path: Path, exclude: Set):
        """
        Dumps vocabularies indices to json files

        Parameters
        ----------
        path: Path
            Path to the directory where the files will be saved
        exclude: Set
            The set of component names to exclude from saving
            This is useful when components are repeated in the pipeline.
        """
        if self.name in exclude:
            return
        exclude.add(self.name)

        self.embedding.save_extra_data(path / "embedding", exclude)

        os.makedirs(path, exist_ok=True)

        with (path / "label_voc.json").open("w") as f:
            json.dump(self.label_voc.indices, f)

    def load_extra_data(self, path: Path, exclude: Set):
        """
        Loads vocabularies indices from json files

        Parameters
        ----------
        path: Path
            Path to the directory where the files will be loaded
        exclude: Set
            The set of component names to exclude from loading
            This is useful when components are repeated in the pipeline.
        """

        if self.name in exclude:
            return
        exclude.add(self.name)

        self.embedding.load_extra_data(path / "embedding", exclude)

        label_voc_indices = dict(self.label_voc.indices)

        with (path / "label_voc.json").open("r") as f:
            self.label_voc.indices = json.load(f)

        self.update_weights_from_vocab_(label_voc_indices)
