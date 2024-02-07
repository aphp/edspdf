import json
import os
from pathlib import Path
from typing import Set

import regex
import torch
from anyascii import anyascii
from foldedtensor import FoldedTensor, as_folded_tensor
from typing_extensions import TypedDict

from edspdf import TrainablePipe
from edspdf.layers.vocabulary import Vocabulary
from edspdf.pipeline import Pipeline
from edspdf.pipes.embeddings import EmbeddingOutput
from edspdf.registry import registry
from edspdf.structures import PDFDoc

BoxTextEmbeddingInputBatch = TypedDict(
    "BoxTextEmbeddingInputBatch",
    {
        # "mask": torch.BoolTensor,
        "tokens_shape": FoldedTensor,
        "tokens_prefix": FoldedTensor,
        "tokens_suffix": FoldedTensor,
        "tokens_norm": FoldedTensor,
    },
)


def word_shape(text: str) -> str:
    """
    Converts a word into its shape following the algorithm used in the
    spaCy library.

    https://github.com/explosion/spaCy/blob/b69d249a/spacy/lang/lex_attrs.py#L118

    Parameters
    ----------
    text: str

    Returns
    -------
    str
    The word shape
    """
    if len(text) >= 100:
        return "LONG"

    shape = []
    last = ""
    seq = 0
    for char in text:
        if char.isalpha():
            if char.isupper():
                shape_char = "X"
            else:
                shape_char = "x"
        elif char.isdigit():
            shape_char = "d"
        else:
            shape_char = char
        if shape_char == last:
            seq += 1
        else:
            seq = 0
            last = shape_char
        if seq < 4:
            shape.append(shape_char)
    return "".join(shape)


@registry.factory.register("simple-text-embedding")
class SimpleTextEmbedding(TrainablePipe[EmbeddingOutput]):
    """
    A module that embeds the textual features of the blocks
    """

    def __init__(
        self,
        size: int,
        pipeline: Pipeline = None,
        name: str = "simple-text-embedding",
    ):
        """
        Parameters
        ----------
        size: int
            Size of the output box embedding
        pipeline: Pipeline
            The pipeline object
        name: str
            Name of the component
        """
        super().__init__(pipeline, name)

        self.size = size
        self.shape_voc = Vocabulary(["__unk__"], default=0)
        self.prefix_voc = Vocabulary(["__unk__"], default=0)
        self.suffix_voc = Vocabulary(["__unk__"], default=0)
        self.norm_voc = Vocabulary(["__unk__"], default=0)

        self.shape_embedding = torch.nn.Embedding(len(self.shape_voc), self.size)
        self.prefix_embedding = torch.nn.Embedding(len(self.prefix_voc), self.size)
        self.suffix_embedding = torch.nn.Embedding(len(self.suffix_voc), self.size)
        self.norm_embedding = torch.nn.Embedding(len(self.norm_voc), self.size)

        punct = "[:punct:]" + "\"'ˊ＂〃ײ᳓″״‶˶ʺ“”˝"
        num_like = r"\d+(?:[.,]\d+)?"
        default = rf"[^\d{punct}'\n[[:space:]]+(?:['ˊ](?=[[:alpha:]]|$))?"
        self.word_regex = regex.compile(
            rf"({num_like}|[{punct}]|[\n\r\t]|[^\S\r\n\t]+|{default})([^\S\r\n\t])?"
        )

    @property
    def output_size(self):
        return self.size

    def update_weights_from_vocab_(self, vocab_items_before):
        for name in ["shape", "prefix", "suffix", "norm"]:
            embedding = getattr(self, f"{name}_embedding")
            vocab = getattr(self, f"{name}_voc")
            items_before = vocab_items_before[name]

            old_index = [items_before[item] for item in items_before]
            new_index = [vocab.indices[item] for item in items_before]
            new_embedding = torch.nn.Embedding(
                num_embeddings=len(vocab.indices),
                embedding_dim=embedding.embedding_dim,
            )
            new_embedding.weight.data[new_index] = embedding.weight.data[old_index]
            embedding.weight = new_embedding.weight
            embedding.num_embeddings = len(vocab.indices)

    def post_init(self, gold_data, exclude: set):
        if self.name in exclude:
            return

        exclude.add(self.name)
        vocab_items_before = {
            "shape": dict(self.shape_voc.indices),
            "prefix": dict(self.prefix_voc.indices),
            "suffix": dict(self.suffix_voc.indices),
            "norm": dict(self.norm_voc.indices),
        }

        shape_init = self.shape_voc.initialization()
        prefix_init = self.prefix_voc.initialization()
        suffix_init = self.suffix_voc.initialization()
        norm_init = self.norm_voc.initialization()
        with shape_init, prefix_init, suffix_init, norm_init:  # noqa: E501
            for doc in gold_data:
                self.preprocess(doc)

        self.update_weights_from_vocab_(vocab_items_before)

    def to_disk(self, path: Path, exclude: Set):
        if self.name in exclude:
            return

        exclude.add(self.name)
        os.makedirs(path, exist_ok=True)
        with (path / "shape_voc.json").open("w") as f:
            json.dump(self.shape_voc.indices, f)
        with (path / "prefix_voc.json").open("w") as f:
            json.dump(self.prefix_voc.indices, f)
        with (path / "suffix_voc.json").open("w") as f:
            json.dump(self.suffix_voc.indices, f)
        with (path / "norm_voc.json").open("w") as f:
            json.dump(self.norm_voc.indices, f)

        return super().to_disk(path, exclude)

    def from_disk(self, path: Path, exclude: Set):
        if self.name in exclude:
            return

        exclude.add(self.name)
        vocab_items_before = {
            "shape": dict(self.shape_voc.indices),
            "prefix": dict(self.prefix_voc.indices),
            "suffix": dict(self.suffix_voc.indices),
            "norm": dict(self.norm_voc.indices),
        }
        with (path / "shape_voc.json").open("r") as f:
            self.shape_voc.indices = json.load(f)
        with (path / "prefix_voc.json").open("r") as f:
            self.prefix_voc.indices = json.load(f)
        with (path / "suffix_voc.json").open("r") as f:
            self.suffix_voc.indices = json.load(f)
        with (path / "norm_voc.json").open("r") as f:
            self.norm_voc.indices = json.load(f)

        self.update_weights_from_vocab_(vocab_items_before)

        super().from_disk(path, exclude)

    def preprocess(self, doc: PDFDoc):
        tokens_shape = []
        tokens_prefix = []
        tokens_suffix = []
        tokens_norm = []

        for page in doc.pages:
            text_boxes = page.text_boxes
            tokens_shape.append([[] for _ in text_boxes])
            tokens_prefix.append([[] for _ in text_boxes])
            tokens_suffix.append([[] for _ in text_boxes])
            tokens_norm.append([[] for _ in text_boxes])

            for i, b in enumerate(text_boxes):
                words = [m.group(0) for m in self.word_regex.finditer(b.text)]

                for word in words:
                    # ascii_str = unidecode.unidecode(word)
                    ascii_str = anyascii(word).strip()
                    tokens_shape[-1][i].append(
                        self.shape_voc.encode(word_shape(ascii_str))
                    )
                    tokens_prefix[-1][i].append(
                        self.prefix_voc.encode(ascii_str.lower()[:3])
                    )
                    tokens_suffix[-1][i].append(
                        self.suffix_voc.encode(ascii_str.lower()[-3:])
                    )
                    tokens_norm[-1][i].append(self.norm_voc.encode(ascii_str.lower()))

        return {
            "tokens_shape": tokens_shape,
            "tokens_prefix": tokens_prefix,
            "tokens_suffix": tokens_suffix,
            "tokens_norm": tokens_norm,
        }

    def collate(self, batch) -> BoxTextEmbeddingInputBatch:
        kwargs = dict(
            dtype=torch.long,
            data_dims=("word",),
            full_names=(
                "sample",
                "page",
                "line",
                "word",
            ),
        )

        return {
            "tokens_shape": as_folded_tensor(batch["tokens_shape"], **kwargs),
            "tokens_prefix": as_folded_tensor(batch["tokens_prefix"], **kwargs),
            "tokens_suffix": as_folded_tensor(batch["tokens_suffix"], **kwargs),
            "tokens_norm": as_folded_tensor(batch["tokens_norm"], **kwargs),
        }

    def forward(self, batch: BoxTextEmbeddingInputBatch) -> EmbeddingOutput:
        text_embeds = (
            self.shape_embedding(batch["tokens_shape"].as_tensor())
            + self.prefix_embedding(batch["tokens_prefix"].as_tensor())
            + self.suffix_embedding(batch["tokens_suffix"].as_tensor())
            # + self.norm_embedding(batch["tokens_norm"].as_tensor())
        )

        return {"embeddings": batch["tokens_shape"].with_data(text_embeds)}
