from enum import Enum

import torch

from edspdf import Module, registry

from .box_preprocessor import BoxPreprocessor
from .sinusoidal_embedding import SinusoidalEmbedding


class PositionEmbeddingMode(str, Enum):
    sin = "sin"
    learned = "learned"


@registry.factory.register("box-embedding")
class BoxEmbedding(Module):
    def __init__(
        self,
        size: int,
        n_positions: int,
        x_mode: PositionEmbeddingMode = "sin",
        y_mode: PositionEmbeddingMode = "sin",
        w_mode: PositionEmbeddingMode = "sin",
        h_mode: PositionEmbeddingMode = "sin",
    ):
        """
        Encodes a box using its geometrical features, as extracted by the
        BoxPreprocessor module.

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

        self.box_preprocessor = BoxPreprocessor()
        self.x_embedding = self._make_embedding(n_positions, size // 6, x_mode)
        self.y_embedding = self._make_embedding(n_positions, size // 6, y_mode)
        self.w_embedding = self._make_embedding(n_positions, size // 6, w_mode)
        self.h_embedding = self._make_embedding(n_positions, size // 6, h_mode)
        self.first_page_embedding = torch.nn.Parameter(torch.randn(self.size))
        self.last_page_embedding = torch.nn.Parameter(torch.randn(self.size))
        self.preprocess = self.box_preprocessor.preprocess
        self.collate = self.box_preprocessor.collate

    @classmethod
    def _make_embedding(cls, n_positions, size, mode):
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
