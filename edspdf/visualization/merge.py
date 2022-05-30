from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.linalg import block_diag


def sort_lines(lines: pd.DataFrame) -> pd.DataFrame:
    sorted_lines = lines.sort_values(["page", "label"])
    sorted_lines["key"] = range(len(lines))
    return sorted_lines


class Comparable(BaseModel, arbitrary_types_allowed=True):
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray


def build_comparison(
    lines: pd.DataFrame, page: int, label: str
) -> Tuple[Comparable, Comparable]:
    df = lines.query("page == @page")

    comparable = df.query("label == @label")
    others = df.query("label != @label")

    comparable_obj = Comparable(
        x0=comparable.x0.values,
        x1=comparable.x1.values,
        y0=comparable.y0.values,
        y1=comparable.y1.values,
    )

    others_obj = Comparable(
        x0=others.x0.values,
        x1=others.x1.values,
        y0=others.y0.values,
        y1=others.y1.values,
    )

    return comparable_obj, others_obj


def iter_comparison(lines: pd.DataFrame) -> Iterable[Tuple[Comparable, Comparable]]:

    df = lines[["page", "label"]].drop_duplicates()

    for page, label in zip(df.page, df.label):
        yield build_comparison(lines=lines, page=page, label=label)


def compute_sub_adj(comparable: Comparable, others: Comparable) -> np.ndarray:

    if not len(others.x0):
        return np.ones((len(comparable.x0), len(comparable.x0))) > 0

    X0 = np.minimum(comparable.x0[None, :], comparable.x0[:, None])
    X1 = np.maximum(comparable.x1[None, :], comparable.x1[:, None])
    Y0 = np.minimum(comparable.y0[None, :], comparable.y0[:, None])
    Y1 = np.maximum(comparable.y1[None, :], comparable.y1[:, None])

    DX = np.minimum(X1[:, :, None], others.x1[None, None, :])
    DX -= np.maximum(X0[:, :, None], others.x0[None, None, :])

    DY = np.minimum(Y1[:, :, None], others.y1[None, None, :])
    DY -= np.maximum(Y0[:, :, None], others.y0[None, None, :])

    adj = np.logical_or(DX <= 0, DY <= 0).all(axis=2)

    return adj


def build_full_adj_matrix(lines: pd.DataFrame) -> np.ndarray:
    adjs = []

    for c, o in iter_comparison(lines=lines):
        adjs.append(compute_sub_adj(c, o))

    return block_diag(*adjs)


def compute_clique(adj: np.ndarray) -> List[List[int]]:

    G = nx.from_numpy_array(adj)

    return list(nx.find_cliques(G))


def process_cliques(
    lines: pd.DataFrame, cliques: List[List[int]]
) -> Tuple[pd.DataFrame, bool]:

    seen = set()
    has_overlap = False

    cliques = sorted(cliques, key=len, reverse=True)

    for i, clique in enumerate(cliques):
        for element in clique:
            if element in seen:
                has_overlap = True
            else:
                seen.add(element)
                lines["key"].iat[element] = i

    return lines, has_overlap


def merge_lines(lines: pd.DataFrame) -> pd.DataFrame:

    if len(lines) == 0:
        return lines

    lines["lab"] = lines["label"]

    while True:
        lines = sort_lines(lines).copy()

        adj = build_full_adj_matrix(lines)
        cliques = compute_clique(adj)

        lines, has_overlap = process_cliques(lines, cliques)

        if not has_overlap:
            break

        lines["label"] = lines["key"]

    merged = lines.groupby(["key"]).agg(
        page=("page", "first"),
        x0=("x0", "min"),
        y0=("y0", "min"),
        x1=("x1", "max"),
        y1=("y1", "max"),
        label=("lab", "first"),
    )

    return merged
