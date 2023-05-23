from enum import Enum
from typing import TypeVar

import torch

Args = TypeVar("Args")


def pad_2d(data, pad=0, device=None):
    max_len = max(map(len, data), default=0)
    padded = [row + [pad] * (max_len - len(row)) for row in data]
    return torch.as_tensor(padded, device=device)


def compute_pdf_relative_positions(x0, y0, x1, y1, width, height, n_relative_positions):
    """
    Compute relative positions between boxes.
    Input boxes must be split between pages with the shape n_pages * n_boxes

    Parameters
    ----------
    x0: torch.FloatTensor
    y0: torch.FloatTensor
    x1: torch.FloatTensor
    y1: torch.FloatTensor
    width: torch.FloatTensor
    height: torch.FloatTensor
    n_relative_positions: int
        Maximum range of embeddable relative positions between boxes (further
        distances will be capped to ±n_relative_positions // 2)

    Returns
    -------
    torch.LongTensor
        Shape: n_pages * n_boxes * n_boxes * 2
    """
    dx = x0[:, None, :] - x0[:, :, None]  # B begin -> A begin
    dx = (dx * n_relative_positions).long()

    dy = y0[:, None, :] - y0[:, :, None]
    # If query above (dy > 0) key, use query height
    ref_height = (dy >= 0).float() * height.float()[:, :, None] + (
        dy < 0
    ).float() * height[:, None, :]
    dy0 = y1[:, None, :] - y0[:, :, None]  # A begin -> B end
    dy1 = y0[:, None, :] - y1[:, :, None]  # A end -> B begin
    offset = 0.5
    dy = torch.where(
        # where A fully above B (dy0 and dy1 > 0), dy is min distance
        ((dy0 + offset).sign() > 0) & ((dy1 + offset).sign() > 0),
        (torch.minimum(dy0, dy1) / ref_height + offset).ceil(),
        # where A fully below B (dy0 and dy1 < 0), dy is -(min -distances)
        torch.where(
            ((dy0 - offset).sign() < 0) & ((dy1 - offset).sign() < 0),
            (torch.maximum(dy0, dy1) / ref_height - offset).floor(),
            0,
        ),
    )
    dy = (dy.abs().ceil() * dy.sign()).long()

    relative_positions = torch.stack([dx, dy], dim=-1)

    return relative_positions


class ActivationFunction(str, Enum):
    relu = "relu"
    gelu = "gelu"
    glu = "glu"


def get_activation_function(activation: ActivationFunction):
    return getattr(torch.nn.functional, activation)


def log_einsum_exp(formula, *ops):
    """
    Numerically stable log of einsum of exponents of operands
    """
    maxes = [op.max() for op in ops]
    ops = [op - op_max for op, op_max in zip(ops, maxes)]
    res = torch.einsum(formula, *(op.exp() for op in ops)).log()
    res = res + sum(maxes)
    return res


def convert_flattened_to_padded(
    inputs: torch.FloatTensor,
    lengths: torch.LongTensor,
    return_mask: bool = False,
    return_nesting_indices: bool = False,
):
    n_samples, seq_size, dim = inputs.shape
    lengths_mask = lengths != -1
    keep_list = lengths_mask.view(-1).tolist()
    device = inputs.device

    flat_offsets = (
        lengths.cumsum(1).masked_fill(~lengths_mask, -1)
        + seq_size * torch.arange(n_samples, device=device).unsqueeze(1)
    ).view(-1)

    embedding_views = torch.tensor_split(inputs.view(-1, dim), flat_offsets)[:-1]
    embedding_padded = torch.nn.utils.rnn.pad_sequence(
        [x for x, keep in zip(embedding_views, keep_list) if keep],
        batch_first=True,
    )

    nesting_indices = lengths_mask.view(-1).long().cumsum(0).roll(1)
    nesting_indices[0] = 0
    nesting_indices = nesting_indices.view(lengths_mask.shape)

    if return_mask:
        positions = torch.arange(embedding_padded.size(1), device=device)
        mask_padded = positions.unsqueeze(0) < lengths[lengths_mask].unsqueeze(1)
        if return_nesting_indices:
            return embedding_padded, nesting_indices, mask_padded
        else:
            return embedding_padded, mask_padded
    if return_nesting_indices:
        return embedding_padded, nesting_indices
    return embedding_padded
