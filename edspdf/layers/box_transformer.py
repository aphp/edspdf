from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from foldedtensor import FoldedTensor
from typing_extensions import Literal

from edspdf.layers.relative_attention import RelativeAttention
from edspdf.utils.torch import (
    ActivationFunction,
    compute_pdf_relative_positions,
    get_activation_function,
)


class BoxTransformerLayer(torch.nn.Module):
    """
    BoxTransformerLayer combining a self attention layer and a
    linear->activation->linear transformation. This layer is used in the
    [BoxTransformerModule][edspdf.layers.box_transformer.BoxTransformerModule] module.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int = 2,
        dropout_p: float = 0.0,
        head_size: Optional[int] = None,
        activation: ActivationFunction = "gelu",
        init_resweight: float = 0.0,
        attention_mode: Sequence[Literal["c2c", "c2p", "p2c"]] = ("c2c", "c2p", "p2c"),
        position_embedding: Optional[
            Union[torch.FloatTensor, torch.nn.Parameter]
        ] = None,
    ):
        """
        Parameters
        ----------
        input_size: int
            Input embedding size
        num_heads: int
            Number of attention heads in the attention layer
        dropout_p: float
            Dropout probability both for the attention layer and embedding projections
        head_size: int
            Head sizes of the attention layer
        activation: ActivationFunction
            Activation function used in the linear->activation->linear transformation
        init_resweight: float
            Initial weight of the residual gates.
            At 0, the layer acts (initially) as an identity function, and at 1 as
            a standard Transformer layer.
            Initializing with a value close to 0 can help the training converge.
        attention_mode: Sequence[Literal["c2c", "c2p", "p2c"]]
            Mode of relative position infused attention layer.
            See the
            [relative attention][edspdf.layers.relative_attention.RelativeAttention]
            documentation for more information.
        position_embedding: torch.FloatTensor
            Position embedding to use as key/query position embedding in the attention
            computation.
        """
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout_p)

        self.attention = RelativeAttention(
            size=input_size,
            n_heads=num_heads,
            do_pooling=True,
            head_size=head_size,
            position_embedding=position_embedding,
            dropout_p=dropout_p,
            n_coordinates=2,
            mode=attention_mode,
        )
        self.resweight = torch.nn.Parameter(torch.Tensor([float(init_resweight)]))
        self.norm = torch.nn.LayerNorm(input_size)
        self.activation = get_activation_function(activation)

        self.resweight2 = torch.nn.Parameter(torch.Tensor([float(init_resweight)]))
        self.norm2 = torch.nn.LayerNorm(input_size)
        self.linear1 = torch.nn.Linear(input_size, input_size * 2)
        self.linear2 = torch.nn.Linear(input_size * 2, input_size)

    def forward(
        self,
        embeds: torch.FloatTensor,
        mask: torch.BoolTensor,
        relative_positions: torch.LongTensor,
        no_position_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass of the BoxTransformerLayer

        Parameters
        ----------
        embeds: torch.FloatTensor
            Embeddings to contextualize
            Shape: `n_samples * n_keys * input_size`
        mask: torch.BoolTensor
            Mask of the embeddings. 0 means padding element.
            Shape: `n_samples * n_keys`
        relative_positions: torch.LongTensor
            Position of the keys relatively to the query elements
            Shape: `n_samples * n_queries * n_keys * n_coordinates (2 for x/y)`
        no_position_mask: Optional[torch.BoolTensor]
            Key / query pairs for which the position attention terms should
            be disabled.
            Shape: `n_samples * n_queries * n_keys`

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            - Contextualized embeddings
              Shape: `n_samples * n_queries * n_keys`
            - Attention logits
              Shape: `n_samples * n_queries * n_keys * n_heads`
        """
        update, attn = self.attention(
            embeds,
            embeds,
            embeds,
            mask,
            relative_positions=relative_positions,
            no_position_mask=no_position_mask,
        )
        embeds = embeds + self.dropout(update) * self.resweight
        embeds = self.norm(embeds)

        update = self.linear2(self.dropout(self.activation(self.linear1(embeds))))
        embeds = embeds + self.dropout(update) * self.resweight2
        embeds = self.norm2(embeds)

        return embeds, attn


class BoxTransformerModule(torch.nn.Module):
    def __init__(
        self,
        input_size: Optional[int] = None,
        num_heads: int = 2,
        dropout_p: float = 0.0,
        head_size: Optional[int] = None,
        activation: ActivationFunction = "gelu",
        init_resweight: float = 0.0,
        n_relative_positions: Optional[int] = None,
        attention_mode: Sequence[Literal["c2c", "c2p", "p2c"]] = ("c2c", "c2p", "p2c"),
        n_layers: int = 2,
    ):
        """
        Box Transformer architecture combining a multiple
        [BoxTransformerLayer][edspdf.layers.box_transformer.BoxTransformerLayer]
        modules. It is mainly used in
        [BoxTransformer][edspdf.pipes.embeddings.box_transformer.BoxTransformer].

        Parameters
        ----------
        input_size: int
            Input embedding size
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
        attention_mode: Sequence[Literal["c2c", "c2p", "p2c"]]
            Mode of relative position infused attention layer.
            See the
            [relative attention][edspdf.layers.relative_attention.RelativeAttention]
            documentation for more information.
        n_layers: int
            Number of layers in the Transformer
        """

        super().__init__()

        self.n_relative_positions = n_relative_positions
        self.dropout = torch.nn.Dropout(dropout_p)
        self.empty_embed = torch.nn.Parameter(torch.randn(input_size))
        position_embedding = (
            torch.nn.Parameter(
                torch.randn(
                    (
                        n_relative_positions,
                        input_size,
                    )
                )
            )
            if n_relative_positions is not None
            else None
        )
        self.layers = torch.nn.ModuleList(
            [
                BoxTransformerLayer(
                    input_size=input_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    dropout_p=dropout_p,
                    activation=activation,
                    init_resweight=init_resweight,
                    attention_mode=attention_mode,
                    position_embedding=position_embedding,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        embeds: FoldedTensor,
        boxes: Dict,
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Forward pass of the BoxTransformer

        Parameters
        ----------
        embeds: torch.FloatTensor
            Embeddings to contextualize
            Shape: `n_samples * n_keys * input_size`
        boxes: Dict
            Layout features of the input elements

        Returns
        -------
        Tuple[torch.FloatTensor, List[torch.FloatTensor]]
            - Output of the last BoxTransformerLayer
              Shape: `n_samples * n_queries * n_keys`
            - Attention logits of all layers
              Shape: `n_samples * n_queries * n_keys * n_heads`
        """
        embeds = embeds.refold("page", "line")
        mask = embeds.mask
        data = embeds.as_tensor()

        n_pages, seq_size, dim = data.shape
        device = data.device
        data_with_cls = torch.cat(
            [
                self.empty_embed * torch.ones(n_pages, 1, 1, device=device),
                data,
            ],
            dim=1,
        )
        mask_with_cls = torch.cat(
            [
                torch.ones(n_pages, 1, dtype=torch.bool, device=device),
                mask,
            ],
            dim=1,
        )
        n = seq_size + 1
        relative_positions = None
        no_position_mask = None
        if self.n_relative_positions is not None:
            relative_positions = torch.zeros(
                n_pages, n, n, 2, dtype=torch.long, device=device
            )
            relative_positions[:, 1:, 1:, :] = compute_pdf_relative_positions(
                x0=boxes["xmin"].refold("page", "line"),
                x1=boxes["xmax"].refold("page", "line"),
                y0=boxes["ymin"].refold("page", "line"),
                y1=boxes["ymax"].refold("page", "line"),
                width=boxes["width"].refold("page", "line"),
                height=boxes["height"].refold("page", "line"),
                n_relative_positions=self.n_relative_positions,
            )
            no_position_mask = torch.ones(
                n_pages, n, n, dtype=torch.bool, device=device
            )
            no_position_mask[:, 1:, 1:] = 0

        attention = []
        for layer in self.layers:
            data_with_cls, attn = layer(
                data_with_cls,
                mask_with_cls,
                relative_positions=relative_positions,
                no_position_mask=no_position_mask,
            )
            attention.append(attn)

        return embeds.with_data(data_with_cls[:, 1:])
