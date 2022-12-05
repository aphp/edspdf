from typing import List, Optional, Sequence, Tuple, Union

import torch

from edspdf.utils.torch import ActivationFunction, get_activation_function

from .relative_attention import RelativeAttention, RelativeAttentionMode


class BlockTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int = 2,
        dropout_p: float = 0.15,
        head_size: Optional[int] = None,
        activation: ActivationFunction = "gelu",
        init_resweight: float = 0.0,
        attention_mode: Sequence[RelativeAttentionMode] = ("c2c", "c2p", "p2c"),
        position_embedding: Optional[torch.FloatTensor] = None,
    ):
        """
        BlockTransformerLayer combining a self attention layer and a
        linear->activation->linear transformation.

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
        Forward pass of the BlockTransformerLayer

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


class BlockTransformer(torch.nn.Module):
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
            Union[torch.nn.Parameter, torch.FloatTensor]
        ] = None,
        n_layers: int = 2,
    ):
        """
        BlockTransformer combining a multiple BlockTransformerLayer

        Parameters
        ----------
        input_size: int
            Input embedding size
        num_heads: int
            Number of attention heads in the attention layers
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
        position_embedding: torch.FloatTensor
            Position embedding to use as key/query position embedding in the attention
            computation.
        n_layers: int
            Number of layers in the Transformer
        """

        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.position_embedding = position_embedding
        self.layers = torch.nn.ModuleList(
            [
                BlockTransformerLayer(
                    input_size=input_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    dropout_p=dropout_p,
                    activation=activation,
                    init_resweight=init_resweight,
                    attention_mode=attention_mode,
                    position_embedding=self.position_embedding,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        embeds: torch.FloatTensor,
        mask: torch.BoolTensor,
        relative_positions: torch.LongTensor,
        no_position_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Forward pass of the BlockTransformer

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
        Tuple[torch.FloatTensor, List[torch.FloatTensor]]
        - Output of the last BlockTransformerLayer
          Shape: `n_samples * n_queries * n_keys`
        - Attention logits of all layers
          Shape: `n_samples * n_queries * n_keys * n_heads`
        """
        attention = []
        for layer in self.layers:
            embeds, attn = layer(
                embeds,
                mask,
                relative_positions=relative_positions,
                no_position_mask=no_position_mask,
            )
            attention.append(attn)

        return embeds, attention
