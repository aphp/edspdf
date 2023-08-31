from typing import Any, Sequence, TypeVar

import numpy as np

from ..structures import Box
from .collections import list_factorize

INF = 100000

T = TypeVar("T", bound=Box)


def _align_box_labels_on_page(
    src_boxes: Sequence[Box],
    dst_boxes: Sequence[Box],
    threshold: float = 0.0001,
    pollution_label: Any = None,
):
    if len(src_boxes) == 0 or len(dst_boxes) == 0:
        return []

    src_labels, label_vocab = list_factorize(
        [b.label for b in src_boxes] + [pollution_label]
    )
    src_labels = np.asarray(src_labels)

    src_x0, src_x1, src_y0, src_y1 = np.asarray(
        [(b.x0, b.x1, b.y0, b.y1) for b in src_boxes] + [(-INF, INF, -INF, INF)]
    ).T[:, :, None]
    dst_x0, dst_x1, dst_y0, dst_y1 = np.asarray(
        [(b.x0, b.x1, b.y0, b.y1) for b in dst_boxes]
    ).T[:, None, :]

    # src_x0 has shape (n_src_boxes, 1)
    # dst_x0 has shape (1, n_dst_boxes)

    dx = np.minimum(src_x1, dst_x1) - np.maximum(src_x0, dst_x0)  # shape: n_src, n_dst
    dy = np.minimum(src_y1, dst_y1) - np.maximum(src_y0, dst_y0)  # shape: n_src, n_dst

    overlap = np.clip(dx, 0, None) * np.clip(dy, 0, None)  # shape: n_src, n_dst
    src_area = (src_x1 - src_x0) * (src_y1 - src_y0)  # shape: n_src
    dst_area = (dst_x1 - dst_x0) * (dst_y1 - dst_y0)  # shape: n_dst

    # To remove errors for 0 divisions
    src_area[src_area == 0] = 1
    dst_area[dst_area == 0] = 1

    covered_src_ratio = overlap / src_area  # shape: n_src, n_dst
    covered_dst_ratio = overlap / dst_area  # shape: n_src, n_dst

    score = covered_src_ratio
    score[covered_dst_ratio < threshold] = 0.0

    src_indices = score.argmax(0)
    dst_labels = src_labels[src_indices]

    new_dst_boxes = [
        b.evolve(label=label_vocab[label_idx])
        for b, label_idx in zip(dst_boxes, dst_labels)
        # if label_vocab[label_idx] != "__pollution__"
    ]
    return new_dst_boxes


def align_box_labels(
    src_boxes: Sequence[Box],
    dst_boxes: Sequence[T],
    threshold: float = 0.0001,
    pollution_label: Any = None,
) -> Sequence[T]:
    """
    Align lines with possibly overlapping (and non-exhaustive) labels.

    Possible matches are sorted by covered area. Lines with no overlap at all

    Parameters
    ----------
    src_boxes: Sequence[Box]
        The labelled boxes that will be used to determine the label of the dst_boxes
    dst_boxes: Sequence[T]
        The non-labelled boxes that will be assigned a label
    threshold : float, default 1
        Threshold to use for discounting a label. Used if the `labels` DataFrame
        does not provide a `threshold` column, or to fill `NaN` values thereof.
    pollution_label : Any
        The label to use for boxes that are not covered by any of the source boxes

    Returns
    -------
    List[Box]
        A copy of the boxes, with the labels mapped from the source boxes
    """

    return [
        b
        for page in sorted(set((b.page_num for b in dst_boxes)))
        for b in _align_box_labels_on_page(
            src_boxes=[
                b
                for b in src_boxes
                if page is None or b.page_num is None or b.page_num == page
            ],
            dst_boxes=[
                b
                for b in dst_boxes
                if page is None or b.page_num is None or b.page_num == page
            ],
            threshold=threshold,
            pollution_label=pollution_label,
        )
    ]
