from typing import Any, Dict

import regex
import torch
from unidecode import unidecode

from edspdf import TrainableComponent, registry
from edspdf.utils.collections import flatten
from edspdf.utils.torch import pad_2d

from .cnn_encoder import CnnPooler
from .vocabulary import Vocabulary


def word_shape(text: str) -> str:
    """
    Converts a word into its shape

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


@registry.factory.register("text-embedding")
class TextEmbedding(TrainableComponent):
    def __init__(
        self,
        size: int,
        pooler: Dict[str, Any] = {},
    ):
        """

        Parameters
        ----------
        size: int
            Size of the output box embedding
        pooler: Dict
            The module used to encode the textual features of the blocks
        """
        super().__init__()

        assert size % 6 == 0, "Embedding dimension must be dividable by 6"

        self.size = size
        self.shape_voc = Vocabulary(["__unk__"], default=0)
        self.prefix_voc = Vocabulary(["__unk__"], default=0)
        self.suffix_voc = Vocabulary(["__unk__"], default=0)
        self.norm_voc = Vocabulary(["__unk__"], default=0)

        self.shape_embedding = None
        self.prefix_embedding = None
        self.suffix_embedding = None
        self.norm_embedding = None
        self.hpos_embedding = None
        self.vpos_embedding = None
        self.first_page_embedding = None
        self.last_page_embedding = None

        self.pooler = CnnPooler(
            input_size=size,
            output_size=size,
            **pooler,
        )

        punct = "[:punct:]" + "\"'ˊ＂〃ײ᳓″״‶˶ʺ“”˝"
        num_like = r"\d+(?:[.,]\d+)?"
        default = rf"[^\d{punct}'\n[[:space:]]+(?:['ˊ](?=[[:alpha:]]|$))?"
        self.word_regex = regex.compile(
            rf"({num_like}|[{punct}]|[\n\r\t]|[^\S\r\n\t]+|{default})([^\S\r\n\t])?"
        )

    @property
    def output_size(self):
        return self.size

    def initialize(self, gold_data):
        shape_init = self.shape_voc.initialization()
        prefix_init = self.prefix_voc.initialization()
        suffix_init = self.suffix_voc.initialization()
        norm_init = self.norm_voc.initialization()
        with shape_init, prefix_init, suffix_init, norm_init:  # noqa: E501
            with self.no_cache():
                for doc in gold_data:
                    self.preprocess(doc, supervision=True)

        self.shape_embedding = torch.nn.Embedding(len(self.shape_voc), self.size)
        self.prefix_embedding = torch.nn.Embedding(len(self.prefix_voc), self.size)
        self.suffix_embedding = torch.nn.Embedding(len(self.suffix_voc), self.size)
        self.norm_embedding = torch.nn.Embedding(len(self.norm_voc), self.size)

    def preprocess(self, doc, supervision: bool = False):
        text_boxes = doc.lines

        tokens_shape = [[] for _ in text_boxes]
        tokens_prefix = [[] for _ in text_boxes]
        tokens_suffix = [[] for _ in text_boxes]
        tokens_norm = [[] for _ in text_boxes]
        for i, b in enumerate(text_boxes):
            words = [m.group(0) for m in self.word_regex.finditer(b.text)]

            for word in words:
                ascii_str = unidecode(word)
                tokens_shape[i].append(self.shape_voc.encode(word_shape(ascii_str)))
                tokens_prefix[i].append(self.prefix_voc.encode(ascii_str.lower()[:3]))
                tokens_suffix[i].append(self.suffix_voc.encode(ascii_str.lower()[-3:]))
                tokens_norm[i].append(self.norm_voc.encode(ascii_str.lower()))

        return {
            "tokens_shape": tokens_shape,
            "tokens_prefix": tokens_prefix,
            "tokens_suffix": tokens_suffix,
            "tokens_norm": tokens_norm,
        }

    def collate(self, batch, device: torch.device):
        self.last_prep = batch

        # print("Collating embedding")

        shapes = pad_2d(flatten(batch["tokens_shape"]), pad=-1, device=device)
        mask = shapes != -1
        shapes[~mask] = 0
        return {
            "tokens_shape": shapes,
            "tokens_prefix": pad_2d(
                flatten(batch["tokens_prefix"]), pad=0, device=device
            ),
            "tokens_suffix": pad_2d(
                flatten(batch["tokens_suffix"]), pad=0, device=device
            ),
            "tokens_norm": pad_2d(flatten(batch["tokens_norm"]), pad=0, device=device),
            "tokens_mask": mask,
        }

    def forward(self, batch, supervision=False):
        self.last_batch = batch

        text_embeds = self.pooler(
            embeds=(
                self.shape_embedding(batch["tokens_shape"])
                + self.prefix_embedding(batch["tokens_prefix"])
                + self.suffix_embedding(batch["tokens_suffix"])
                # + self.norm_embedding(batch["tokens_norm"])
            ),
            mask=batch["tokens_mask"],
        )

        return text_embeds
