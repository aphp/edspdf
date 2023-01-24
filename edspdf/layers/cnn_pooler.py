from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from edspdf import Module, registry
from edspdf.utils.torch import ActivationFunction, get_activation_function


@registry.factory.register("cnn-pooler")
class CnnPooler(Module):
    """
    One dimension CNN encoding multi-kernel layer.
    Input embeddings are convoluted using linear kernels each parametrized with
    a (window) size of `kernel_size[kernel_i]`
    The output of the kernels are concatenated together, max-pooled and finally
    projected to a size of `output_size`.
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        activation: ActivationFunction = "relu",
    ):
        """
        Parameters
        ----------
        input_size: int
            Size of the input embeddings
        output_size: Optional[int]
            Size of the output embeddings
            Defaults to the `input_size`
        out_channels: int
            Number of channels
        kernel_sizes: Sequence[int]
            Window size of each kernel
        activation: str
            Activation function to use
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.activation = get_activation_function(activation)

    def initialize(self, gold_data, **kwargs):
        super().initialize(gold_data, **kwargs)

        if self.out_channels is None:
            self.out_channels = self.input_size
        if self.output_size is None:
            self.output_size = self.input_size

        self.convolutions = torch.nn.ModuleList(
            torch.nn.Conv1d(
                in_channels=self.input_size,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                padding=0,
            )
            for kernel_size in self.kernel_sizes
        )
        self.linear = torch.nn.Linear(
            in_features=self.out_channels * len(self.kernel_sizes),
            out_features=self.output_size,
        )

    def forward(self, embeds: torch.FloatTensor, mask: torch.BoolTensor):
        """
        Encode embeddings with a 1d convolutional network

        Parameters
        ----------
        embeds: torch.FloatTensor
            Input embeddings
            Shape: `n_samples * n_elements * input_size`
        mask: torch.BoolTensor
            Input mask. 0 values are for padding elements.
            Padding elements are masked with a 0 value before the convolution.
            Shape: `n_samples * n_elements * input_size`

        Returns
        -------
        torch.FloatTensor
        Shape: `n_samples * output_size`
        """
        embeds = embeds.masked_fill(~mask.unsqueeze(-1), 0).permute(
            0, 2, 1
        )  # sample word dim -> sample dim word
        embeds = torch.cat(
            [
                self.activation(
                    # pad by the appropriate amount on both sides of each sentence
                    conv(
                        F.pad(
                            embeds,
                            pad=[
                                conv.kernel_size[0] // 2,
                                (conv.kernel_size[0] - 1) // 2,
                            ],
                        )
                    )
                    .permute(0, 2, 1)
                    .masked_fill(~mask.unsqueeze(-1), 0)
                )
                for conv in self.convolutions
            ],
            dim=2,
        )
        pooled = embeds.max(1).values

        return self.linear(pooled)
