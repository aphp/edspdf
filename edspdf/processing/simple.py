from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

from edspdf.utils.collections import batchify, flatten

if TYPE_CHECKING:
    from edspdf.lazy_collection import LazyCollection

batch_size_fns = {
    "content_boxes": lambda batch: sum(len(doc.content_boxes) for doc in batch),
    "pages": lambda batch: sum(len(doc.pages) for doc in batch),
    "docs": len,
}

doc_size_fns = {
    "content_boxes": lambda doc: len(doc.content_boxes),
}


def apply_basic_pipes(docs, pipes):
    for name, pipe, kwargs in pipes:
        if hasattr(pipe, "batch_process"):
            docs = pipe.batch_process(docs)
        else:
            docs = [pipe(doc, **kwargs) for doc in docs]
    return docs


def execute_simple_backend(
    lc: LazyCollection,
):
    """
    This is the default execution mode which batches the documents and processes each
    batch on the current process in a sequential manner.
    """
    try:
        no_grad = sys.modules["torch"].no_grad
    except (KeyError, AttributeError):
        no_grad = nullcontext
    reader = lc.reader
    writer = lc.writer
    show_progress = lc.show_progress

    split_into_batches_after = lc.split_into_batches_after
    if split_into_batches_after is None or lc.batch_by != "docs" or lc.sort_chunks:
        split_into_batches_after = next(
            (p[0] for p in lc.pipeline if p[0] is not None), None
        )
    names = [step[0] for step in lc.pipeline] + [None]
    chunk_components = lc.pipeline[: names.index(split_into_batches_after)]
    batch_components = lc.pipeline[names.index(split_into_batches_after) :]

    def process():
        bar = nullcontext()
        if show_progress:
            from tqdm import tqdm

            bar = tqdm(smoothing=0.1, mininterval=5.0)

        with bar:
            for docs in batchify(
                (
                    subtask
                    for task, count in reader.read_main()
                    for subtask in reader.read_worker([task])
                ),
                batch_size=lc.chunk_size,
            ):
                docs = apply_basic_pipes(docs, chunk_components)

                if lc.sort_chunks:
                    docs.sort(
                        key=doc_size_fns.get(
                            lc.sort_chunks, doc_size_fns["content_boxes"]
                        )
                    )

                batches = [
                    batch
                    for batch in batchify(
                        docs,
                        batch_size=lc.batch_size,
                        formula=batch_size_fns.get(lc.batch_by, len),
                    )
                ]

                for batch in batches:
                    with no_grad(), lc.cache():
                        batch = apply_basic_pipes(batch, batch_components)

                    if writer is not None:
                        result, count = writer.write_worker(batch)
                        if show_progress:
                            bar.update(count)
                        yield result
                    else:
                        if show_progress:
                            bar.update(len(batch))
                        yield batch
            if writer is not None:
                result, count = writer.finalize()
                if show_progress:
                    bar.update(count)
                if count:
                    yield result

    gen = process()
    return flatten(gen) if writer is None else writer.write_main(gen)
