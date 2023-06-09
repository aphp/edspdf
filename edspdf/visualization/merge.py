from typing import List, Sequence

import networkx as nx
import numpy as np

from edspdf.structures import Box

INF = 1000000


def merge_boxes(
    boxes: Sequence[Box],
) -> List[Box]:
    """
    Recursively merge boxes that have the same label to form larger non-overlapping
    boxes.

    Parameters
    ----------
    boxes: Sequence[Box]
        List of boxes to merge

    Returns
    -------
    List[Box]
        List of merged boxes
    """
    labels = np.asarray([b.label for b in boxes])

    coords = np.asarray([(b.x0, b.x1, b.y0, b.y1) for b in boxes])

    # Key that determines if two boxes can be merged, initialized from the box labels
    merge_keys = np.unique(labels, return_inverse=True)[1]

    # For each page
    while True:
        adj = np.zeros((len(boxes), len(boxes)), dtype=bool)

        # Split boxes between those that belong to a label (and could be merged),
        # and those that do not belong to that label and will prevent the mergers
        for key in np.unique(merge_keys):
            key_filter = merge_keys == key

            x0, x1, y0, y1 = coords[key_filter].T
            obs_x0, obs_x1, obs_y0, obs_y1 = coords[~key_filter].T

            A = (slice(None), None, None)
            B = (None, slice(None), None)

            # Find the bbox of the hypothetical merged boxes
            merged_x0 = np.minimum(x0[A], x0[B])
            merged_x1 = np.maximum(x1[A], x1[B])
            merged_y0 = np.minimum(y0[A], y0[B])
            merged_y1 = np.maximum(y1[A], y1[B])

            # And detect if it overlaps existing box of a different label
            dx = np.minimum(merged_x1, obs_x1) - np.maximum(merged_x0, obs_x0)
            dy = np.minimum(merged_y1, obs_y1) - np.maximum(merged_y0, obs_y0)
            merged_overlap_with_other = (dx > 0) & (dy > 0)
            no_box_inbetween = (~merged_overlap_with_other).all(-1)

            # Update the adjacency matrix to 1 if two boxes can be merged
            # (ie no box of a different label lie inbetween)
            adj_indices = np.flatnonzero(key_filter)
            adj[adj_indices[:, None], adj_indices[None, :]] = no_box_inbetween

        # Build the cliques of boxes that can be merged
        cliques = nx.find_cliques(nx.from_numpy_array(adj))

        # These cliques of mergeable boxes can be overlapping: think of a cross
        # like this=
        # *** --- ***
        # --- --- ---
        # *** --- ***
        # for which the two (-) labelled cliques would be the two axis of the cross
        # For each box, we change its label to its first clique number, so the cross
        # looks like this (symbols between the 2 figures don't map to the same indices)
        # *** --- ***
        # ooo ooo ooo
        # *** --- ***
        # and rerun the above process until there is no conflict

        conflicting_cliques = False
        seen = set()
        for clique_idx, clique_box_indices in enumerate(cliques):
            for box_idx in clique_box_indices:
                if box_idx in seen:
                    # print("Already seen", box_idx)
                    conflicting_cliques = True
                else:
                    seen.add(box_idx)
                    merge_keys[box_idx] = clique_idx

        if not conflicting_cliques:
            break

    x0, x1, y0, y1 = coords.T.reshape((4, -1))

    # Finally, compute the bbox of the sets of mergeable boxes (same `key`)
    merged_boxes = []
    for group_key in dict.fromkeys(merge_keys):
        indices = [i for i, key in enumerate(merge_keys) if group_key == key]
        first_box = boxes[indices[0]]
        merged_boxes.append(
            first_box.evolve(
                x0=min(x0[i] for i in indices),
                y0=min(y0[i] for i in indices),
                x1=max(x1[i] for i in indices),
                y1=max(y1[i] for i in indices),
            )
        )

    return merged_boxes
