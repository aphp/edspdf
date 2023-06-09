from foldedtensor import FoldedTensor
from typing_extensions import TypedDict

from edspdf import TrainablePipe

EmbeddingOutput = TypedDict(
    "EmbeddingOutput",
    {
        "embeddings": FoldedTensor,
    },
)
