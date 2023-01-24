from enum import Enum
from typing import Iterable, Optional

import torch

from edspdf import Module, registry

from .box_layout_preprocessor import BoxLayoutPreprocessor
from .sinusoidal_embedding import SinusoidalEmbedding


class PositionEmbeddingMode(str, Enum):
    sin = "sin"
    learned = "learned"


@registry.factory.register("box-layout-embedding")
class BoxLayoutEmbedding(Module):
    """
    Encodes a box using its geometrical features, as extracted by the
    BoxLayoutPreprocessor module.
    """

    def __init__(
        self,
        n_positions: int,
        size: Optional[int] = None,
        x_mode: PositionEmbeddingMode = "sin",
        y_mode: PositionEmbeddingMode = "sin",
        w_mode: PositionEmbeddingMode = "sin",
        h_mode: PositionEmbeddingMode = "sin",
    ):
        """

        Parameters
        ----------
        size: int
            Size of the output box embedding
        n_positions: int
            Number of position embeddings stored in the PositionEmbedding module
        x_mode: PositionEmbeddingMode
            Position embedding mode of the x coordinates
        y_mode: PositionEmbeddingMode
            Position embedding mode of the x coordinates
        w_mode: PositionEmbeddingMode
            Position embedding mode of the width features
        h_mode: PositionEmbeddingMode
            Position embedding mode of the height features
        """

        super().__init__()

        self.n_positions = n_positions
        self.size = size
        self.x_mode = x_mode
        self.y_mode = y_mode
        self.w_mode = w_mode
        self.h_mode = h_mode
        self.x_embedding = None
        self.y_embedding = None
        self.w_embedding = None
        self.h_embedding = None
        self.first_page_embedding = None
        self.last_page_embedding = None

        self.box_preprocessor = BoxLayoutPreprocessor()
        self.preprocess = self.box_preprocessor.preprocess
        self.collate = self.box_preprocessor.collate

    def initialize(self, gold_data: Iterable, size: int = None, **kwargs):
        super().initialize(gold_data, size=size, **kwargs)
        n_pos, size = self.n_positions, self.size

        self.x_embedding = self._make_embed(n_pos, size // 6, self.x_mode)
        self.y_embedding = self._make_embed(n_pos, size // 6, self.y_mode)
        self.w_embedding = self._make_embed(n_pos, size // 6, self.w_mode)
        self.h_embedding = self._make_embed(n_pos, size // 6, self.h_mode)
        self.first_page_embedding = torch.nn.Parameter(torch.randn(self.size))
        self.last_page_embedding = torch.nn.Parameter(torch.randn(self.size))

    @classmethod
    def _make_embed(cls, n_positions, size, mode):
        if mode == "sin":
            return SinusoidalEmbedding(n_positions, size)
        else:
            return torch.nn.Embedding(n_positions, size)

    def forward(self, batch):
        return (
            torch.cat(
                [
                    self.x_embedding(
                        (batch["xmin"] * self.n_positions)
                        .clamp(max=self.n_positions - 1)
                        .long()
                    ),
                    self.y_embedding(
                        (batch["ymin"] * self.n_positions)
                        .clamp(max=self.n_positions - 1)
                        .long()
                    ),
                    self.x_embedding(
                        (batch["xmax"] * self.n_positions)
                        .clamp(max=self.n_positions - 1)
                        .long()
                    ),
                    self.y_embedding(
                        (batch["ymax"] * self.n_positions)
                        .clamp(max=self.n_positions - 1)
                        .long()
                    ),
                    self.w_embedding(
                        (batch["width"] * self.n_positions)
                        .clamp(max=self.n_positions - 1)
                        .long()
                    ),
                    self.h_embedding(
                        (batch["height"] * 5 * self.n_positions)
                        .clamp(max=self.n_positions - 1)
                        .long()
                    ),
                ],
                dim=-1,
            )
            + self.first_page_embedding * batch["first_page"][..., None]
            + self.last_page_embedding * batch["last_page"][..., None]
        )
