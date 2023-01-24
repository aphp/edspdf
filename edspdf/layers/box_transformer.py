from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from edspdf.utils.torch import (
    ActivationFunction,
    compute_pdf_relative_positions,
    get_activation_function,
)

from .. import Module, registry
from .relative_attention import RelativeAttention, RelativeAttentionMode


class BoxTransformerLayer(torch.nn.Module):
    """
    BoxTransformerLayer combining a self attention layer and a
    linear->activation->linear transformation.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int = 2,
        dropout_p: float = 0.15,
        head_size: Optional[int] = None,
        activation: ActivationFunction = "gelu",
        init_resweight: float = 0.0,
        attention_mode: Sequence[RelativeAttentionMode] = ("c2c", "c2p", "p2c"),
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
        attention_mode: Sequence[RelativeAttentionMode]
            Mode of relative position infused attention layer.
            See the [relative attention](relative_attention) documentation for more
            information.
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


@registry.factory.register("box-transformer")
class BoxTransformer(Module):
    def __init__(
        self,
        input_size: Optional[int] = None,
        num_heads: int = 2,
        dropout_p: float = 0.15,
        head_size: Optional[int] = None,
        activation: ActivationFunction = "gelu",
        init_resweight: float = 0.0,
        n_relative_positions: Optional[int] = None,
        attention_mode: Sequence[RelativeAttentionMode] = ("c2c", "c2p", "p2c"),
        n_layers: int = 2,
    ):
        """
        BoxTransformer combining a multiple BoxTransformerLayer

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
        attention_mode: Sequence[RelativeAttentionMode]
            Mode of relative position infused attention layer.
            See the [relative attention](relative_attention) documentation for more
            information.
        n_layers: int
            Number of layers in the Transformer
        """

        super().__init__()

        self.dropout = torch.nn.Dropout(dropout_p)
        self.input_size = input_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_size = head_size
        self.activation = activation
        self.init_resweight = init_resweight
        self.n_relative_positions = n_relative_positions
        self.attention_mode = attention_mode
        self.n_layers = n_layers

    def initialize(self, gold_data: Iterable, input_size: int = None, **kwargs):
        super().initialize(gold_data, input_size=input_size, **kwargs)

        self.empty_embed = torch.nn.Parameter(torch.randn(self.input_size))
        self.position_embedding = torch.nn.Parameter(
            torch.randn(
                (
                    self.n_relative_positions,
                    self.input_size,
                )
            )
        )
        self.layers = torch.nn.ModuleList(
            [
                BoxTransformerLayer(
                    input_size=self.input_size,
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                    dropout_p=self.dropout_p,
                    activation=self.activation,
                    init_resweight=self.init_resweight,
                    attention_mode=self.attention_mode,
                    position_embedding=self.position_embedding,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(
        self,
        embeds: torch.FloatTensor,
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

        page_boxes = boxes["page_ids"]
        mask = page_boxes != -1
        n_samples, n_boxes_per_sample = mask.shape

        n_pages = page_boxes.shape[0]
        device = page_boxes.device
        embeds_with_cls = torch.cat(
            [
                self.empty_embed * torch.ones(n_pages, 1, 1, device=device),
                embeds[page_boxes],
            ],
            dim=1,
        )
        mask_with_cls = torch.cat(
            [
                torch.ones(n_samples, 1, dtype=torch.bool, device=device),
                mask,
            ],
            dim=1,
        )
        n = n_boxes_per_sample + 1
        relative_positions = torch.zeros(
            n_pages, n, n, 2, dtype=torch.long, device=device
        )
        relative_positions[:, 1:, 1:, :] = compute_pdf_relative_positions(
            x0=boxes["xmin"][page_boxes],
            x1=boxes["xmax"][page_boxes],
            y0=boxes["ymin"][page_boxes],
            y1=boxes["ymax"][page_boxes],
            width=boxes["width"][page_boxes],
            height=boxes["height"][page_boxes],
            n_relative_positions=self.n_relative_positions,
        )
        no_position_mask = torch.ones(n_pages, n, n, dtype=torch.bool, device=device)
        no_position_mask[:, 1:, 1:] = 0

        attention = []
        for layer in self.layers:
            embeds, attn = layer(
                embeds_with_cls,
                mask_with_cls,
                relative_positions=relative_positions,
                no_position_mask=no_position_mask,
            )
            attention.append(attn)

        return embeds[:, 1:][mask]
