import torch
from typing_extensions import Literal

from edspdf.layers.sinusoidal_embedding import SinusoidalEmbedding
from edspdf.pipeline import Pipeline
from edspdf.pipes.embeddings import EmbeddingOutput
from edspdf.pipes.embeddings.box_layout_preprocessor import (
    BoxLayoutBatch,
    BoxLayoutPreprocessor,
)
from edspdf.registry import registry
from edspdf.trainable_pipe import TrainablePipe


@registry.factory.register("box-layout-embedding")
class BoxLayoutEmbedding(TrainablePipe[EmbeddingOutput]):
    """
    This component encodes the geometrical features of a box, as extracted by the
    BoxLayoutPreprocessor module, into an embedding. For position modes, use:

    - `"sin"` to embed positions with a fixed
      [SinusoidalEmbedding][edspdf.layers.sinusoidal_embedding.SinusoidalEmbedding]
    - `"learned"` to embed positions using a learned standard pytorch embedding layer

    Each produces embedding is the concatenation of the box width, height and the top,
    left, bottom and right coordinates, each embedded depending on the `*_mode` param.

    Parameters
    ----------
    size: int
        Size of the output box embedding
    n_positions: int
        Number of position embeddings stored in the PositionEmbedding module
    x_mode: Literal["sin", "learned"]
        Position embedding mode of the x coordinates
    y_mode: Literal["sin", "learned"]
        Position embedding mode of the x coordinates
    w_mode: Literal["sin", "learned"]
        Position embedding mode of the width features
    h_mode: Literal["sin", "learned"]
        Position embedding mode of the height features
    """

    def __init__(
        self,
        n_positions: int,
        size: int,
        x_mode: Literal["sin", "learned"] = "sin",
        y_mode: Literal["sin", "learned"] = "sin",
        w_mode: Literal["sin", "learned"] = "sin",
        h_mode: Literal["sin", "learned"] = "sin",
        pipeline: Pipeline = None,
        name: str = "box-layout-embedding",
    ):
        super().__init__(pipeline, name)

        assert size % 12 == 0, "Size must be a multiple of 12"

        self.n_positions = n_positions
        self.output_size = size

        self.x_embedding = self._make_embed(n_positions, size // 6, x_mode)
        self.y_embedding = self._make_embed(n_positions, size // 6, y_mode)
        self.w_embedding = self._make_embed(n_positions, size // 6, w_mode)
        self.h_embedding = self._make_embed(n_positions, size // 6, h_mode)
        self.first_page_embedding = torch.nn.Parameter(torch.randn(size))
        self.last_page_embedding = torch.nn.Parameter(torch.randn(size))

        self.box_preprocessor = BoxLayoutPreprocessor(pipeline, "box_preprocessor")

    def preprocess(self, doc):
        return self.box_preprocessor.preprocess(doc)

    def collate(self, batch) -> BoxLayoutBatch:
        return self.box_preprocessor.collate(batch)

    @classmethod
    def _make_embed(cls, n_positions, size, mode):
        if mode == "sin":
            return SinusoidalEmbedding(n_positions, size)
        else:
            return torch.nn.Embedding(n_positions, size)

    def forward(self, batch: BoxLayoutBatch) -> EmbeddingOutput:
        # fmt: off
        embedding = (
              torch.cat(
                  [
                      self.x_embedding((batch["xmin"] * self.n_positions).clamp(max=self.n_positions - 1).long()),  # noqa: E501
                      self.y_embedding((batch["ymin"] * self.n_positions).clamp(max=self.n_positions - 1).long()),  # noqa: E501
                      self.x_embedding((batch["xmax"] * self.n_positions).clamp(max=self.n_positions - 1).long()),  # noqa: E501
                      self.y_embedding((batch["ymax"] * self.n_positions).clamp(max=self.n_positions - 1).long()),  # noqa: E501
                      self.w_embedding((batch["width"] * self.n_positions).clamp(max=self.n_positions - 1).long()),  # noqa: E501
                      self.h_embedding((batch["height"] * 5 * self.n_positions).clamp(max=self.n_positions - 1).long()),  # noqa: E501
                  ],
                  dim=-1,
              )
              + self.first_page_embedding * batch["first_page"][..., None]
              + self.last_page_embedding * batch["last_page"][..., None]
        )
        # fmt: on
        return {"embeddings": embedding}
