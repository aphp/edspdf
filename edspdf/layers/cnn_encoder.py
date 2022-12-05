from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from edspdf import registry


@registry.factory.register("cnn-encoder")
class CnnPooler(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 4, 5),
    ):
        """
        One dimension CNN encoding multi-kernel layer.
        Input embeddings are convoluted using linear kernels each parametrized with
        a (window) size of `kernel_size[kernel_i]`
        The output of the kernels are concatenated together, max-pooled and finally
        projected to a size of `output_size`.

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
        """
        super().__init__()
        if output_size is None:
            output_size = input_size
        if out_channels is None:
            out_channels = input_size
        self.convolutions = torch.nn.ModuleList(
            torch.nn.Conv1d(
                in_channels=input_size,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0,
            )
            for kernel_size in kernel_sizes
        )
        self._output_size = out_channels * len(kernel_sizes)
        self.linear = torch.nn.Linear(out_channels * len(kernel_sizes), output_size)

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
        embeds = (
            torch.cat(
                [
                    F.relu(
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
            .max(1)
            .values
        )
        return self.linear(embeds)
