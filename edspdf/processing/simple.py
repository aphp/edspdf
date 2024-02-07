from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

from edspdf.utils.collections import batchify, flatten

if TYPE_CHECKING:
    from edspdf.lazy_collection import LazyCollection


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

    def process():
        bar = nullcontext()
        if show_progress:
            from tqdm import tqdm

            bar = tqdm()

        with bar:
            for batch in batchify(
                (
                    subtask
                    for task, count in reader.read_main()
                    for subtask in reader.read_worker([task])
                ),
                batch_size=lc.batch_size,
            ):
                with no_grad(), lc.cache():
                    for name, pipe, kwargs in lc.pipeline:
                        if hasattr(pipe, "batch_process"):
                            batch = pipe.batch_process(batch, **kwargs)
                        else:
                            batch = [pipe(doc, **kwargs) for doc in batch]

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
