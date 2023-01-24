from typing import Any, Sequence

import numpy as np

from ..models import Box
from .collections import list_factorize

INF = 100000


def _align_box_labels_on_page(
    src_boxes: Sequence[Box],
    dst_boxes: Sequence[Box],
    threshold: float = 0.0001,
    group_by_source: bool = False,
    pollution_label: Any = None,
):
    if len(src_boxes) == 0 or len(dst_boxes) == 0:
        return []

    src_labels, label_vocab = list_factorize(
        [b.label for b in src_boxes] + [pollution_label]
    )
    src_sources, source_vocab = list_factorize(
        [(b.source if b.source is not None else 0) for b in src_boxes],
    )
    src_labels = np.asarray(src_labels)
    src_sources = np.asarray(src_sources)

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

    if not group_by_source:
        src_indices = covered_src_ratio.argmax(0)
        dst_labels = src_labels[src_indices]
    else:
        src_indices, dst_indices = np.flatnonzero(
            (covered_src_ratio > 0) & (covered_dst_ratio > 0)
        )
        dst_labels_scores = np.zeros(
            len(dst_boxes), len(source_vocab), len(label_vocab)
        )
        np.add.at(
            a=dst_labels_scores,
            indices=(dst_indices, src_sources[src_indices], src_labels[src_indices]),
            b=covered_src_ratio[src_indices, dst_indices],
        )
        dst_labels_scores = dst_labels_scores == dst_labels_scores.max(
            -1, keepdims=True
        )
        dst_labels = dst_labels_scores.sum(1).argmax(-1)

    new_dst_boxes = [
        b.evolve(label=label_vocab[label_idx])
        for b, label_idx in zip(dst_boxes, dst_labels)
        # if label_vocab[label_idx] != "__pollution__"
    ]
    return new_dst_boxes


def align_box_labels(
    src_boxes: Sequence[Box],
    dst_boxes: Sequence[Box],
    threshold: float = 0.0001,
    group_by_source: bool = False,
    pollution_label: Any = None,
) -> Sequence[Box]:
    """
    Align lines with possibly overlapping (and non-exhaustive) labels.

    Possible matches are sorted by covered area. Lines with no overlap at all

    Parameters
    ----------
    src_boxes: Sequence[Box]
        The labelled boxes that will be used to determine the label of the dst_boxes
    dst_boxes: Sequence[Box]
        The non-labelled boxes that will be assigned a label
    group_by_source: bool
        Whether to perform majority voting between different sources of
        annotations if any
    threshold : float, default 1
        Threshold to use for discounting a label. Used if the `labels` DataFrame
        does not provide a `threshold` column, or to fill `NaN` values thereof.

    Returns
    -------
    List[Box]
        A copy of the boxes, with the labels mapped from the source boxes
    """

    return [
        b
        for page in sorted(set((b.page for b in dst_boxes)))
        for b in _align_box_labels_on_page(
            src_boxes=[
                b for b in src_boxes if page is None or b.page is None or b.page == page
            ],
            dst_boxes=[
                b for b in dst_boxes if page is None or b.page is None or b.page == page
            ],
            threshold=threshold,
            group_by_source=group_by_source,
            pollution_label=pollution_label,
        )
    ]
