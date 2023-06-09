import math

import torch
import torch.nn.functional as F


class SinusoidalEmbedding(torch.nn.Module):
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
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        temperature: float = 10000.0,
    ):
        """
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
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.temperature = temperature

        weight = torch.zeros(self.num_embeddings, self.embedding_dim)
        position = torch.arange(0, self.num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2).float()
            * (-math.log(self.temperature) / self.embedding_dim)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("weight", weight)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"

    def forward(self, indices: torch.LongTensor):
        """
        Forward pass of the SinusoidalEmbedding module

        Parameters
        ----------
        indices: torch.LongTensor
            Shape: any

        Returns
        -------
        torch.FloatTensor
            Shape: `(*input_shape, embedding_dim)`
        """
        res = F.embedding(indices.clamp(0, len(self.weight) - 1), self.weight)
        return res
