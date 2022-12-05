import torch

from edspdf.utils.torch import pad_2d


def test_pad_2d():
    a = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6],
    ]
    torch.testing.assert_close(
        pad_2d(a, pad=-1),
        torch.tensor(
            [
                [0, 1, 2, 3, 4, -1, -1],
                [0, 1, 2, 3, 4, 5, 6],
            ]
        ),
    )
    torch.testing.assert_close(
        pad_2d([], pad=-1, device=torch.device("cpu")),
        torch.tensor([]),
    )
