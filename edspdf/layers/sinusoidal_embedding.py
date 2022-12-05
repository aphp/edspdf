import math

import torch
import torch.nn.functional as F

from edspdf import registry


@registry.factory.register("sinusoidal-embedding")
class SinusoidalEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        temperature: float = 10000.0,
    ):
        """
        A position embedding lookup table that stores embeddings for a fixed number
        of positions.
        The value of each of the `embedding_dim` channels of the generated embedding
        is generated according to a trigonometric function (sin for even channels,
        cos for odd channels).
        The frequency of the signal in each pair of channels varies according to the
        temperature parameter.

        Any input position above the maximum value `num_embeddings` will be capped to
        `num_embeddings - 1`

        Parameters
        ----------
        num_embeddings: int
            The maximum number of position embeddings store in this table
        embedding_dim: int
            The embedding size
        temperature: float
            The temperature controls the range of frequencies used by each
            channel of the embedding
        """
        torch.nn.Module.__init__(self)

        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2
        self.scale_grad_by_freq = False
        self.sparse = False
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        weight = torch.zeros(num_embeddings, self.embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(temperature) / embedding_dim)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("weight", weight)

    def forward(self, indices: torch.LongTensor):
        """
        Forward pass of the SinusoidalEmbedding module

        Parameters
        ----------
        indices: torch.LongTensor
            Shape: ...

        Returns
        -------
        torch.FloatTensor
        Shape: `... * embedding_dim`
        """
        res = F.embedding(indices.clamp(0, len(self.weight) - 1), self.weight)
        return res
