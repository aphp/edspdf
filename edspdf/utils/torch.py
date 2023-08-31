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
