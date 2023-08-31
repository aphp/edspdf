import math
import operator
from functools import reduce
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import FloatTensor
from torch.nn import Parameter
from typing_extensions import Literal

IMPOSSIBLE = -10000


def make_heads(x, n_heads):
    if isinstance(n_heads, int):
        n_heads = (n_heads,)
    total_heads = reduce(operator.mul, n_heads)
    return x.view(*x.shape[:-1], *n_heads, x.shape[-1] // total_heads)


def gather(tensor, index, dim):
    def arange_at_dim(n, dim, ndim):
        view = [1] * ndim
        view[dim] = -1
        return torch.arange(n, device=tensor.device).view(view)

    dim = (dim + tensor.ndim) % tensor.ndim
    indices = [
        (arange_at_dim(size, i, index.ndim) if i != dim else index)
        for i, size in enumerate(tensor.shape)
    ]
    return tensor[tuple(indices)]


class GroupedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, n_groups=1):
        super().__init__()
        self.n_groups = n_groups
        self.bias = torch.nn.Parameter(torch.zeros(n_groups, output_size))
        self.weight = torch.nn.Parameter(
            torch.stack(
                [
                    torch.nn.Linear(input_size, output_size).weight.data.T
                    for _ in range(n_groups)
                ],
                dim=0,
            )
        )

    def forward(self, x):
        (*base_shape, dim) = x.shape
        x = x.reshape(*base_shape, self.n_groups, dim // self.n_groups)
        x = torch.einsum("...ni,nio->...no", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x.reshape(*base_shape, x.shape[-1] * self.n_groups)


class RelativeAttention(torch.nn.Module):
    """
    A self/cross-attention layer that takes relative position of elements into
    account to compute the attention weights.
    When running a relative attention layer, key and queries are represented using
    content and position embeddings, where position embeddings are retrieved using
    the relative position of keys relative to queries
    """

    def __init__(
        self,
        size: int,
        n_heads: int,
        query_size: Optional[int] = None,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        head_size: Optional[int] = None,
        position_embedding: Optional[Union[FloatTensor, Parameter]] = None,
        dropout_p: float = 0.0,
        same_key_query_proj: bool = False,
        same_positional_key_query_proj: bool = False,
        n_coordinates: int = 1,
        head_bias: bool = True,
        do_pooling: bool = True,
        mode: Sequence[Literal["c2c", "c2p", "p2c"]] = ("c2c", "p2c", "c2p"),
        n_additional_heads: int = 0,
    ):
        """
        Parameters
        ----------
        size: int
            The size of the output embeddings
            Also serves as default if query_size, pos_size, or key_size is None
        n_heads: int
            The number of attention heads
        query_size: Optional[int]
            The size of the query embeddings.
        key_size: Optional[int]
            The size of the key embeddings.
        value_size: Optional[int]
            The size of the value embeddings
        head_size: Optional[int]
            The size of each query / key / value chunk used in the attention dot product
            Default: `key_size / n_heads`
        position_embedding: Optional[torch.FloatTensor]
            The position embedding used as key and query embeddings
        dropout_p: float
            Dropout probability applied on the attention weights
            Default: 0.1
        same_key_query_proj: bool
            Whether to use the same projection operator for content key and queries
            when computing the pre-attention key and query embedding chunks
            Default: False
        same_positional_key_query_proj
            Whether to use the same projection operator for content key and queries
            when computing the pre-attention key and query embedding chunks
            Default: False
        n_coordinates: int
            The number of positional coordinates
            For instance, text is 1D so 1 coordinate, images are 2D so 2 coordinates ...
            Default: 1
        head_bias: bool
            Whether to learn a bias term to add to the attention logits
            This is only useful if you plan to use the attention logits for subsequent
            operations, since attention weights are unaffected by bias terms.
        do_pooling: bool
            Whether to compute the output embedding.
            If you only plan to use attention logits, you should disable this parameter.
            Default: True
        mode: Sequence[Literal["c2c", "c2p", "p2c"]]
            Whether to compute content to content (c2c), content to position (c2p)
            or position to content (p2c) attention terms.
            Setting `mode=('c2c")` disable relative position attention terms: this is
            the standard attention layer.
            To get a better intuition about these different types of attention, here is
            a formulation as fictitious search samples from a word in a (1D) text:

            - content-content : "my content is ’ultrasound’ so I’m looking for other
              words whose content contains information about temporality"
            - content-position: "my content is ’ultrasound’ so I’m looking for other
              words that are 3 positions after of me"
            - position-content : "regardless of my content, I will attend to the word
              one position after from me if it contains information about temporality,
              two words after me if it contains information about location, etc."
        n_additional_heads: int
            The number of additional head logits to compute.
            Those are not used to compute output embeddings, but may be useful in
            subsequent operation.
            Default: 0
        """
        super().__init__()

        if query_size is None:
            query_size = size
        if key_size is None:
            key_size = size
        if value_size is None:
            value_size = key_size
        if head_size is None and key_size is not None:
            assert key_size % n_heads == 0
            head_size = key_size // n_heads
        value_head_size = None
        if do_pooling and size is not None:
            assert size % n_heads == 0
            value_head_size = size // n_heads
        self.n_coordinates = n_coordinates
        self.n_heads = n_heads + n_additional_heads
        self.n_additional_heads = n_additional_heads
        self.mode = mode
        n_query_heads = n_heads + n_additional_heads
        self.content_key_proj = torch.nn.Linear(key_size, n_query_heads * head_size)
        if isinstance(position_embedding, torch.Tensor) and not isinstance(
            position_embedding, torch.nn.Parameter
        ):
            self.register_buffer("position_embedding", position_embedding)
        else:
            self.position_embedding = position_embedding

        if same_key_query_proj:
            self.content_query_proj = self.content_key_proj
        else:
            self.content_query_proj = torch.nn.Linear(
                query_size,
                n_query_heads * head_size,
            )
        if do_pooling:
            self.content_value_proj = torch.nn.Linear(
                value_size, value_head_size * n_heads
            )

        if position_embedding is not None:
            pos_size = self.position_embedding.shape[-1]
            self.position_key_proj = GroupedLinear(
                pos_size // n_coordinates,
                head_size * n_query_heads // n_coordinates,
                n_groups=n_coordinates,
            )
            if same_key_query_proj or same_positional_key_query_proj:
                self.position_query_proj = self.position_key_proj
            else:
                self.position_query_proj = GroupedLinear(
                    pos_size // n_coordinates,
                    head_size * n_query_heads // n_coordinates,
                    n_groups=n_coordinates,
                )

        self.dropout = torch.nn.Dropout(dropout_p)
        if head_bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_query_heads))
        self.output_size = size

    def forward(
        self,
        content_queries: torch.FloatTensor,
        content_keys: Optional[torch.FloatTensor] = None,
        content_values: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.BoolTensor] = None,
        relative_positions: Optional[torch.LongTensor] = None,
        no_position_mask: Optional[torch.BoolTensor] = None,
        base_attn: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]:
        """
        Forward pass of the RelativeAttention layer.

        Parameters
        ----------
        content_queries: torch.FloatTensor
            The content query embedding to use in the attention computation
            Shape: `n_samples * n_queries * query_size`
        content_keys: Optional[torch.FloatTensor]
            The content key embedding to use in the attention computation.
            If None, defaults to the `content_queries`
            Shape: `n_samples * n_keys * query_size`
        content_values: Optional[torch.FloatTensor]
            The content values embedding to use in the final pooling computation.
            If None, pooling won't be performed.
            Shape: `n_samples * n_keys * query_size`
        mask: Optional[torch.BoolTensor]
            The content key embedding to use in the attention computation.
            If None, defaults to the `content_queries`
            Shape: either
            - `n_samples * n_keys`
            - `n_samples * n_queries * n_keys`
            - `n_samples * n_queries * n_keys * n_heads`
        relative_positions: Optional[torch.LongTensor]
            The relative position of keys relative to queries
            If None, positional attention terms won't be computed.
            Shape: `n_samples * n_queries * n_keys * n_coordinates`
        no_position_mask: Optional[torch.BoolTensor]
            Key / query pairs for which the position attention terms should
            be disabled.
            Shape: `n_samples * n_queries * n_keys`
        base_attn: Optional[torch.FloatTensor]
            Attention logits to add to the computed attention logits
            Shape: `n_samples * n_queries * n_keys * n_heads`

        Returns
        -------
        Union[Tuple[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]
            - the output contextualized embeddings (only if content_values is not None
              and the `do_pooling` attribute is set to True)
              Shape: n_sample * n_keys * `size`
            - the attention logits
              Shape: n_sample * n_keys * n_queries * (n_heads + n_additional_heads)
        """
        content_keys = content_queries if content_keys is None else content_keys

        attn = (
            torch.zeros(
                content_queries.shape[0],
                content_queries.shape[1],
                content_keys.shape[1],
                self.n_heads,
                device=content_queries.device,
            )
            if base_attn is None
            else base_attn
        )

        attn_weights = []
        if 0 not in content_queries.shape and 0 not in content_keys.shape:
            content_keys = make_heads(
                self.content_key_proj(self.dropout(content_keys)), self.n_heads
            )
            content_queries = make_heads(
                self.content_query_proj(self.dropout(content_queries)), self.n_heads
            )
            if content_values is not None:
                content_values = make_heads(
                    self.content_value_proj(self.dropout(content_values)),
                    self.n_heads - self.n_additional_heads,
                )

            size = content_queries.shape[-1]
            if "c2c" in self.mode:
                content_to_content_attn = torch.einsum(
                    "nihd,njhd->nijh", content_queries, content_keys
                ) / math.sqrt(size)
                attn_weights.append(content_to_content_attn)

            if relative_positions is not None and (
                "p2c" in self.mode or "c2p" in self.mode
            ):
                position_keys = make_heads(
                    self.position_key_proj(self.dropout(self.position_embedding)),
                    (self.n_coordinates, self.n_heads),
                )
                position_queries = make_heads(
                    self.position_query_proj(self.dropout(self.position_embedding)),
                    (self.n_coordinates, self.n_heads),
                )
                relative_positions = (
                    position_queries.shape[0] // 2 + relative_positions
                ).clamp(0, position_queries.shape[0] - 1)

                if "c2p" in self.mode:
                    content_to_position_attn = torch.einsum(
                        "nihxd,zxhd->nizhx",
                        make_heads(content_queries, self.n_coordinates),
                        position_keys,
                    )
                    content_to_position_attn = gather(
                        content_to_position_attn,
                        index=relative_positions.unsqueeze(-2),
                        dim=2,
                    ).sum(-1) / math.sqrt(size)
                    if no_position_mask is not None:
                        content_to_position_attn = content_to_position_attn.masked_fill(
                            no_position_mask[..., None], 0
                        )
                    attn_weights.append(content_to_position_attn)

                if "p2c" in self.mode:
                    position_to_content_attn = torch.einsum(
                        "zxhd,njhxd->nzjhx",
                        position_queries,
                        make_heads(content_keys, self.n_coordinates),
                    )
                    position_to_content_attn = gather(
                        position_to_content_attn,
                        index=relative_positions.unsqueeze(-2),
                        dim=1,
                    ).sum(-1) / math.sqrt(size)
                    if no_position_mask is not None:
                        position_to_content_attn = position_to_content_attn.masked_fill(
                            no_position_mask[..., None], 0
                        )
                    attn_weights.append(position_to_content_attn)

        attn = attn + sum(attn_weights) / math.sqrt(len(attn_weights))

        if hasattr(self, "bias"):
            attn = attn + self.bias
        if content_values is not None:
            if mask.ndim == 2:
                mask = mask[:, None, :, None]
            if mask.ndim == 3:
                mask = mask[:, :, :, None]

            weights = self.dropout(
                attn[..., self.n_additional_heads :]
                .masked_fill(~mask, IMPOSSIBLE)
                .softmax(-2)
            )
            pooled = torch.einsum("nijh,njhd->nihd", weights, content_values)
            pooled = pooled.reshape(*pooled.shape[:-2], -1)
            return pooled, attn

        return attn
