from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from foldedtensor import as_folded_tensor

from edspdf.pipeline import Pipeline
from edspdf.pipes.embeddings import EmbeddingOutput, TrainablePipe
from edspdf.registry import registry
from edspdf.utils.torch import ActivationFunction, get_activation_function


@registry.factory.register("sub-box-cnn-pooler")
class SubBoxCNNPooler(TrainablePipe[EmbeddingOutput]):
    """
    One dimension CNN encoding multi-kernel layer.
    Input embeddings are convoluted using linear kernels each parametrized with
    a (window) size of `kernel_size[kernel_i]`
    The output of the kernels are concatenated together, max-pooled and finally
    projected to a size of `output_size`.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline instance
    name: str
        Name of the component
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

    def __init__(
        self,
        embedding: TrainablePipe[EmbeddingOutput],
        pipeline: Pipeline = None,
        name: str = "sub-box-cnn-pooler",
        output_size: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        activation: ActivationFunction = "relu",
    ):
        super().__init__(pipeline, name)

        self.activation_fn = get_activation_function(activation)

        self.embedding = embedding
        input_size = self.embedding.output_size
        out_channels = input_size if out_channels is None else out_channels
        output_size = input_size if output_size is None else input_size

        self.convolutions = torch.nn.ModuleList(
            torch.nn.Conv1d(
                in_channels=self.embedding.output_size,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0,
            )
            for kernel_size in kernel_sizes
        )
        self.linear = torch.nn.Linear(
            in_features=out_channels * len(kernel_sizes),
            out_features=output_size,
        )
        self.output_size = output_size

    def forward(self, batch: Any) -> EmbeddingOutput:
        embeddings = self.embedding.module_forward(batch["embedding"])[
            "embeddings"
        ].refold("line", "word")
        if 0 in embeddings.shape:
            return {
                "embeddings": as_folded_tensor(
                    data=torch.zeros(0, self.output_size, device=embeddings.device),
                    lengths=embeddings.lengths[:-1],  # pooled on the last dim
                    data_dims=["line"],  # fully flattened
                    full_names=["sample", "page", "line"],
                )
            }

        # sample word dim -> sample dim word
        box_token_embeddings = embeddings.as_tensor().permute(0, 2, 1)
        box_token_embeddings = torch.cat(
            [
                self.activation_fn(
                    conv(
                        # pad by the appropriate amount on both sides of each sentence
                        F.pad(
                            box_token_embeddings,
                            pad=[
                                conv.kernel_size[0] // 2,
                                (conv.kernel_size[0] - 1) // 2,
                            ],
                        )
                    )
                    .permute(0, 2, 1)
                    .masked_fill(~embeddings.mask.unsqueeze(-1), 0)
                )
                for conv in self.convolutions
            ],
            dim=2,
        )
        pooled = box_token_embeddings.max(1).values
        pooled = self.linear(pooled)
        # print("TEXT EMBEDS", pooled.shape, pooled.sum())

        return {
            "embeddings": as_folded_tensor(
                data=pooled,
                lengths=embeddings.lengths[:-1],  # pooled on the last dim
                data_dims=["line"],  # fully flattened
                full_names=["sample", "page", "line"],
            )
        }
