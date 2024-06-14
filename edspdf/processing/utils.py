import types
from typing import Iterable, List, TypeVar

from edspdf.utils.collections import batchify


def apply_basic_pipes(docs, pipes):
    for name, pipe, kwargs in pipes:
        if hasattr(pipe, "batch_process"):
            docs = pipe.batch_process(docs)
        else:
            results = []
            for doc in docs:
                res = pipe(doc, **kwargs)
                if isinstance(res, types.GeneratorType):
                    results.extend(res)
                else:
                    results.append(res)
            docs = results
    return docs


T = TypeVar("T")


def batchify_with_counts(
    iterable,
    batch_size,
):
    total = 0
    batch = []
    for item, count in iterable:
        if len(batch) > 0 and total + count > batch_size:
            yield batch, total
            batch = []
            total = 0
        batch.append(item)
        total += count
    if len(batch) > 0:
        yield batch, total


def batchify_by_content_boxes(
    iterable: Iterable[T],
    batch_size: int,
    drop_last: bool = False,
) -> Iterable[List[T]]:
    batch = []
    total = 0
    for item in iterable:
        count = len(item.content_boxes)
        if len(batch) > 0 and total + count > batch_size:
            yield batch
            batch = []
            total = 0
        batch.append(item)
        total += count
    if len(batch) > 0 and not drop_last:
        yield batch


def batchify_by_pages(
    iterable: Iterable[T],
    batch_size: int,
    drop_last: bool = False,
) -> Iterable[List[T]]:
    batch = []
    total = 0
    for item in iterable:
        count = len(item.pages)
        if len(batch) > 0 and total + count > batch_size:
            yield batch
            batch = []
            total = 0
        batch.append(item)
        total += count
    if len(batch) > 0 and not drop_last:
        yield batch


batchify_fns = {
    "content_boxes": batchify_by_content_boxes,
    "pages": batchify_by_pages,
    "docs": batchify,
}
