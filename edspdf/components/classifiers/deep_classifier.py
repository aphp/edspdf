from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

from edspdf import Module, TrainableComponent, registry
from edspdf.component import Scorer
from edspdf.layers.vocabulary import Vocabulary
from edspdf.models import PDFDoc
from edspdf.utils.collections import flatten
from edspdf.utils.torch import ActivationFunction, get_activation_function


def classifier_scorer(pairs):
    return classification_report(
        [b.label for pred, gold in pairs for b in gold.lines if b.text != ""],
        [b.label for pred, gold in pairs for b in pred.lines if b.text != ""],
        output_dict=True,
        zero_division=0,
    )


@registry.factory.register("deep-classifier")
class DeepClassifier(TrainableComponent[PDFDoc, Dict[str, Any], PDFDoc]):
    def __init__(
        self,
        embedding: Module,
        labels: Sequence[str] = (),
        activation: ActivationFunction = "gelu",
        dropout_p: float = 0.15,
        scorer: Scorer = classifier_scorer,
    ):
        """
        Runs a deep learning classifier model on the boxes.

        Parameters
        ----------
        labels: Sequence[str]
            Initial labels of the classifier (will be completed during initialization)
        embedding: Module
            Embedding module to encode the PDF boxes
        activation: ActivationFunction
            Name of the activation function
        dropout_p: float
            Dropout probability used on the output of the box and textual encoders
        scorer: Scorer
            Scoring function
        """
        super().__init__()
        self.label_vocabulary: Vocabulary = Vocabulary(
            list(dict.fromkeys(["pollution", *labels]))
        )
        self.embedding: Module = embedding

        size = self.embedding.output_size

        self.linear = torch.nn.Linear(size, size)
        self.classifier: torch.nn.Linear = None  # noqa
        self.activation = get_activation_function(activation)
        self.dropout = torch.nn.Dropout(dropout_p)

        # Scoring function
        self.scorer = scorer

    def initialize(self, gold_data: Iterable[PDFDoc]):
        self.embedding.initialize(gold_data)

        with self.label_vocabulary.initialization():
            for doc in tqdm(gold_data, desc="Initializing classifier"):
                with self.no_cache():
                    self.preprocess(doc, supervision=True)

        self.classifier = torch.nn.Linear(
            in_features=self.embedding.output_size,
            out_features=len(self.label_vocabulary),
        )

    def preprocess(self, doc: PDFDoc, supervision: bool = False) -> Dict[str, Any]:
        result = {
            "embedding": self.embedding.preprocess(doc, supervision=supervision),
            "doc_id": doc.id,
        }
        if supervision:
            text_boxes = doc.lines
            result["labels"] = [
                self.label_vocabulary.encode(b.label) if b.label is not None else -100
                for b in text_boxes
            ]
        return result

    def collate(self, batch, device: torch.device) -> Dict:
        collated = {
            "embedding": self.embedding.collate(batch["embedding"], device),
            "doc_id": batch["doc_id"],
        }
        if "labels" in batch:
            collated.update(
                {
                    "labels": torch.as_tensor(flatten(batch["labels"]), device=device),
                }
            )

        return collated

    def forward(self, batch: Dict, supervision=False) -> Dict:
        embeds = self.embedding(batch["embedding"])

        output = {"loss": 0}

        # Label prediction / learning
        logits = self.classifier(embeds)
        if supervision:
            targets = batch["labels"]
            output["label_loss"] = F.cross_entropy(
                logits,
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
            (b for doc in docs for b in doc.lines),
            output["labels"].cpu().tolist(),
        ):
            if b.text == "":
                b.label = None
            else:
                b.label = self.label_vocabulary.decode(label)
        return docs
