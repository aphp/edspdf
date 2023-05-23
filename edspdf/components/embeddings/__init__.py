import torch
from foldedtensor import FoldedTensor
from typing_extensions import TypedDict

from edspdf import TorchComponent

EmbeddingOutput = TypedDict(
    "EmbeddingOutput",
    {
        "embeddings": FoldedTensor,
    },
)

BoxEmbeddingComponent = TorchComponent[EmbeddingOutput]
