from typing import Optional, Sequence

from typing_extensions import Literal, TypedDict

from edspdf import TrainablePipe
from edspdf.layers.box_transformer import BoxTransformerModule
from edspdf.pipeline import Pipeline
from edspdf.pipes.embeddings import EmbeddingOutput
from edspdf.pipes.embeddings.box_layout_preprocessor import (
    BoxLayoutBatch,
    BoxLayoutPreprocessor,
)
from edspdf.registry import registry
from edspdf.utils.torch import ActivationFunction

BoxTransformerEmbeddingInputBatch = TypedDict(
    "BoxTransformerEmbeddingInputBatch",
    {
        "embedding": EmbeddingOutput,
        "box_prep": BoxLayoutBatch,
    },
)


@registry.factory.register("box-transformer")
class BoxTransformer(TrainablePipe[EmbeddingOutput]):
    """
    BoxTransformer using
    [BoxTransformerModule][edspdf.layers.box_transformer.BoxTransformerModule]
    under the hood.

    !!! note

        This module is a [TrainablePipe][edspdf.trainable_pipe.TrainablePipe]
        and can be used in a [Pipeline][edspdf.pipeline.Pipeline], while
        [BoxTransformerModule][edspdf.layers.box_transformer.BoxTransformerModule]
        is a standard PyTorch module, which does not take care of the
        preprocessing, collating, etc. of the input documents.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline instance
    name: str
        Name of the component
    num_heads: int
        Number of attention heads in the attention layers
    n_relative_positions: int
        Maximum range of embeddable relative positions between boxes (further
        distances are capped to ±n_relative_positions // 2)
    dropout_p: float
        Dropout probability both for the attention layers and embedding projections
    head_size: int
        Head sizes of the attention layers
    activation: ActivationFunction
        Activation function used in the linear->activation->linear transformations
    init_resweight: float
        Initial weight of the residual gates.
        At 0, the layer acts (initially) as an identity function, and at 1 as
        a standard Transformer layer.
        Initializing with a value close to 0 can help the training converge.
    attention_mode: Sequence[RelativeAttentionMode]
        Mode of relative position infused attention layer.
        See the [relative attention][edspdf.layers.relative_attention.RelativeAttention]
        documentation for more information.
    n_layers: int
        Number of layers in the Transformer
    """

    def __init__(
        self,
        embedding: TrainablePipe[EmbeddingOutput],
        num_heads: int = 2,
        dropout_p: float = 0.0,
        head_size: Optional[int] = None,
        activation: ActivationFunction = "gelu",
        init_resweight: float = 0.0,
        n_relative_positions: Optional[int] = None,
        attention_mode: Sequence[Literal["c2c", "c2p", "p2c"]] = ("c2c", "c2p", "p2c"),
        n_layers: int = 2,
        pipeline: Pipeline = None,
        name: str = "box-transformer",
    ):
        super().__init__(pipeline, name)
        self.embedding = embedding
        self.transformer = BoxTransformerModule(
            input_size=embedding.output_size,
            num_heads=num_heads,
            dropout_p=dropout_p,
            head_size=head_size,
            activation=activation,
            init_resweight=init_resweight,
            n_relative_positions=n_relative_positions,
            attention_mode=attention_mode,
            n_layers=n_layers,
        )
        self.output_size = embedding.output_size
        self.box_prep = BoxLayoutPreprocessor(pipeline, f"{name}.box_prep")

    def forward(
        self,
        batch: BoxTransformerEmbeddingInputBatch,
    ) -> EmbeddingOutput:
        res = self.embedding.module_forward(batch["embedding"])
        assert (
            "lengths" not in res
        ), "You must pool a SubBoxEmbedding output before using BoxTransformer"
        return {
            "embeddings": self.transformer(res["embeddings"], batch["box_prep"]),
        }
