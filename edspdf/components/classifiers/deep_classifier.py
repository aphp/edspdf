from typing import Any, Dict, Iterable, NewType, Sequence

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

from edspdf import Module, TrainableComponent, registry
from edspdf.component import Scorer
from edspdf.layers.box_preprocessor import BoxPreprocessor
from edspdf.layers.relative_attention import RelativeAttention, RelativeAttentionMode
from edspdf.layers.vocabulary import Vocabulary
from edspdf.models import PDFDoc
from edspdf.utils.collections import flatten
from edspdf.utils.torch import (
    ActivationFunction,
    compute_pdf_relative_positions,
    get_activation_function,
    log_einsum_exp,
)

ClassifierBatch = NewType("ClassifierBatch", dict)


IMPOSSIBLE = -100000.0


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
        labels: Sequence[str],
        embedding: Module,
        activation: ActivationFunction = "gelu",
        do_harmonize: bool = True,
        n_relative_positions: int = 64,
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
        do_harmonize: bool
            Perform the harmonization process
        n_relative_positions: int
            Maximum range of embeddable relative positions between boxes during the
            harmonization process
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

        self.do_harmonize = do_harmonize
        if self.do_harmonize:
            self.n_relative_positions = n_relative_positions
            self.box_preprocessor = BoxPreprocessor()

            self.adjacency = RelativeAttention(
                size=self.embedding.output_size,
                n_heads=2,
                do_pooling=False,
                head_size=self.embedding.output_size,
                position_embedding=torch.nn.Parameter(
                    torch.randn((n_relative_positions, self.embedding.output_size))
                ),
                dropout_p=0,
                n_coordinates=2,
                mode=(
                    RelativeAttentionMode.c2c,
                    RelativeAttentionMode.c2p,
                    RelativeAttentionMode.p2c,
                ),
            )

        self.linear = torch.nn.Linear(size, size)
        self.classifier: torch.nn.Linear = None  # noqa
        self.activation = get_activation_function(activation)
        self.dropout = torch.nn.Dropout(dropout_p)

        # Scoring function
        self.scorer = scorer

    def initialize(self, gold_data: Iterable[PDFDoc]):
        self.embedding.initialize(gold_data)

        with self.label_vocabulary.initialization():
            for doc in gold_data:
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
        if self.do_harmonize:
            result["boxes"] = self.box_preprocessor.preprocess(
                doc, supervision=supervision
            )
        if supervision:
            text_boxes = doc.lines
            result["labels"] = [
                self.label_vocabulary.encode(b.label) if b.label is not None else -100
                for b in text_boxes
            ]
            if self.do_harmonize:
                next_indices = [-1] * len(text_boxes)
                prev_indices = [-1] * len(text_boxes)
                for i, b in enumerate(doc.lines):
                    if b.next_box is not None:
                        next_i = doc.lines.index(b.next_box)
                        next_indices[i] = next_i
                        prev_indices[next_i] = i
                result["next_indices"] = next_indices
                result["prev_indices"] = prev_indices
        return result

    def collate(self, batch, device: torch.device) -> Dict:
        self.last_prep = batch

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
        if "next_indices" in batch:
            shifted_next_indices = []
            shifted_prev_indices = []
            for doc_next_indices in batch["next_indices"]:
                offset = len(shifted_next_indices)
                shifted_next_indices.extend(
                    [i + offset if i > 0 else -1 for i in doc_next_indices]
                )
            for doc_prev_indices in batch["prev_indices"]:
                offset = len(shifted_prev_indices)
                shifted_prev_indices.extend(
                    [i + offset if i > 0 else -1 for i in doc_prev_indices]
                )
            collated.update(
                {
                    "next_indices": torch.as_tensor(
                        shifted_next_indices, device=device
                    ),
                    "prev_indices": torch.as_tensor(
                        shifted_prev_indices, device=device
                    ),
                }
            )
        if "boxes" in batch:
            collated.update(
                {
                    "boxes": self.box_preprocessor.collate(batch["boxes"], device),
                }
            )

        return collated

    def harmonize(self, block_label_logits, next_logits, prev_logits):
        n_pages, n_blocks, n_labels = block_label_logits.shape
        device = block_label_logits.device

        constraints = torch.eye(n_labels, n_labels + 1, device=device)
        constraints[:, -1] = 1

        is_next_logits = next_logits.log_softmax(-1)
        is_prev_logits = prev_logits.log_softmax(-1)

        label_logits = torch.full(
            (n_pages, n_blocks + 1, n_labels + 1), IMPOSSIBLE, device=device
        )
        label_logits[:n_pages, :n_blocks, :n_labels] = block_label_logits
        label_logits[:, -1, -1] = 0
        label_logits = updated_label_logits = label_logits.log_softmax(-1)
        label_update_logits = torch.zeros_like(label_logits)

        for _ in range(9):
            next_label_logits = log_einsum_exp(
                "pij,pjx->pix",
                is_next_logits,
                updated_label_logits,
            )
            prev_label_logits = log_einsum_exp(
                "pij,pjx->pix",
                is_prev_logits,
                updated_label_logits,
            )
            label_update_logits[..., :n_blocks, :n_labels] = 0.05 * label_update_logits[
                ..., :n_blocks, :n_labels
            ] + 0.95 * (
                log_einsum_exp(
                    "piy,xy->pix",
                    next_label_logits,
                    constraints.log(),
                )
                + log_einsum_exp(
                    "piy,xy->pix",
                    prev_label_logits,
                    constraints.log(),
                )
            )

            updated_label_logits = (label_update_logits + label_logits).log_softmax(-1)

        return updated_label_logits[:n_pages, :n_blocks, :n_labels]

    def forward(self, batch: Dict, supervision=False) -> Dict:
        self.last_batch = batch

        embeds = self.embedding(batch["embedding"])
        device = embeds.device
        # ex: [[0, 2, -1], [1, 3, 4]]

        output = {"loss": 0}

        if self.do_harmonize:
            page_boxes = batch["boxes"]["page_ids"]
            page_boxes_mask = page_boxes != -1

            # n_blocks = embeds.shape[0]

            relative_positions = compute_pdf_relative_positions(
                x0=batch["boxes"]["xmin"][page_boxes],
                x1=batch["boxes"]["xmax"][page_boxes],
                y0=batch["boxes"]["ymin"][page_boxes],
                y1=batch["boxes"]["ymax"][page_boxes],
                width=batch["boxes"]["width"][page_boxes],
                height=batch["boxes"]["height"][page_boxes],
                n_relative_positions=self.n_relative_positions,
            )  # n_pages * n_blocks * n_blocks * 2

            # How to go from initial blocks ([0, 1, 2, 3, 4]) to per-page indice ?
            idx_in_page = page_boxes  # [[0, 2, -1], [1, 3, 4]]
            idx_in_page = idx_in_page.view(-1)  # [0, 2, -1, 1, 3, 4]
            idx_in_page = idx_in_page % len(idx_in_page + 1)  # [0, 2, 5, 1, 3, 4]
            idx_in_page = idx_in_page.argsort()[
                : page_boxes_mask.sum()
            ]  # [0, 3, 1, 4, 5]... and removed [, 2]
            idx_in_page = idx_in_page % page_boxes.shape[-1]  # [0, 0, 1, 1, 2]
            page_range = torch.arange(page_boxes_mask.shape[-1], device=device)
            # During the training, to learn adjacency, the two boxes must exist
            next_mask: torch.BoolTensor = (
                page_boxes_mask[:, :, None]
                & page_boxes_mask[:, None, :]
                # the 1st box must precede the 2dn
                & (page_range[None, :, None] < page_range[None, None, :])
            )
            # During the training, to learn adjacency, the two boxes must exist
            prev_mask: torch.BoolTensor = (
                page_boxes_mask[:, :, None]
                & page_boxes_mask[:, None, :]
                # the 1st box must follow the 2dn
                & (page_range[None, :, None] > page_range[None, None, :])
            )

            next_scores, prev_scores = self.adjacency(
                embeds[page_boxes],
                relative_positions=relative_positions,
            ).unbind(-1)

            next_scores = next_scores.masked_fill(~next_mask, IMPOSSIBLE)
            prev_scores = prev_scores.masked_fill(~prev_mask, IMPOSSIBLE)
            next_scores = torch.cat(
                [next_scores, torch.zeros_like(next_scores[:, :, :1])], dim=2
            )
            prev_scores = torch.cat(
                [prev_scores, torch.zeros_like(prev_scores[:, :, :1])], dim=2
            )

            if supervision:
                labels_per_page = batch["labels"][page_boxes]

                next_target = idx_in_page[
                    batch["next_indices"]
                ]  # n_blocks -> indice in page
                next_target[batch["next_indices"] == -1] = next_scores.shape[-1] - 1
                next_target = next_target[:, None] == torch.arange(
                    next_scores.shape[-1], device=device
                )

                next_target[batch["next_indices"] == -1, :-1] = (
                    (labels_per_page[:, :, None] == labels_per_page[:, None, :])
                    & next_mask
                )[page_boxes_mask][batch["next_indices"] == -1]

                prev_target = idx_in_page[
                    batch["prev_indices"]
                ]  # n_blocks -> indice in page
                prev_target[batch["prev_indices"] == -1] = prev_scores.shape[-1] - 1
                prev_target = prev_target[:, None] == torch.arange(
                    prev_scores.shape[-1], device=device
                )
                prev_target[batch["prev_indices"] == -1, :-1] = (
                    (labels_per_page[:, :, None] == labels_per_page[:, None, :])
                    & prev_mask
                )[page_boxes_mask][batch["prev_indices"] == -1]

                output["adj_loss"] = -(
                    torch.log_softmax(next_scores[page_boxes_mask], dim=-1)
                    .masked_fill(~next_target, -10000)
                    .logsumexp(-1)
                    .sum()
                    + torch.log_softmax(prev_scores[page_boxes_mask], dim=-1)
                    .masked_fill(~prev_target, -10000)
                    .logsumexp(-1)
                    .sum()
                )
                output["loss"] = output["loss"] + output["adj_loss"]

                # block_embeds = block_embeds[page_boxes_mask]

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
            if self.do_harmonize:
                logits = self.harmonize(
                    logits[page_boxes],
                    next_scores,
                    prev_scores,
                )[page_boxes_mask]
            output["labels"] = logits.argmax(-1)
        output["labels_logit"] = logits

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
