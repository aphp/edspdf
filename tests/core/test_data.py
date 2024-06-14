import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

import edspdf
import edspdf.accelerators.multiprocessing
from edspdf import PDFDoc
from edspdf.data.converters import CONTENT, FILENAME
from edspdf.utils.collections import flatten


def box_converter(x):
    return [
        {
            "id": x.id,
            "page_num": b.page_num,
            "x0": b.x0,
            "x1": b.x1,
            "y0": b.y0,
            "y1": b.y1,
        }
        for b in x.content_boxes
    ]


def full_file_converter(x):
    return {
        FILENAME: x.id,
        CONTENT: x.content,
        "annotations": [
            {
                "page_num": b.page_num,
                "x0": b.x0,
                "x1": b.x1,
                "y0": b.y0,
                "y1": b.y1,
            }
            for b in x.content_boxes
        ],
    }


@pytest.mark.parametrize("write_mode", ["parquet", "pandas", "iterable", "files"])
@pytest.mark.parametrize("num_cpu_workers", [1, 2])
@pytest.mark.parametrize("write_in_worker", [False, True])
def test_write_data(
    frozen_pipeline,
    tmp_path,
    change_test_dir,
    write_mode,
    num_cpu_workers,
    write_in_worker,
):
    docs = edspdf.data.read_files("file://" + os.path.abspath("../resources"))
    docs = docs.map_pipeline(frozen_pipeline)
    docs = docs.set_processing(
        num_cpu_workers=num_cpu_workers,
        gpu_pipe_names=[],
        batch_by="content_boxes",
        chunk_size=3,
        sort_chunks=True,
    )
    if write_mode == "parquet":
        docs.write_parquet(
            "file://" + str(tmp_path / "parquet" / "test"),
            converter=box_converter,
            write_in_worker=write_in_worker,
        )
        df = pd.read_parquet("file://" + str(tmp_path / "parquet" / "test"))
    elif write_mode == "pandas":
        if write_in_worker:
            pytest.skip()
        df = docs.to_pandas(converter=box_converter)
    elif write_mode == "iterable":
        if write_in_worker:
            pytest.skip()
        df = pd.DataFrame(flatten(docs.to_iterable(converter=box_converter)))
    else:
        if write_in_worker:
            pytest.skip()
        docs.write_files(
            tmp_path / "files",
            converter=full_file_converter,
        )
        records = []
        for f in (tmp_path / "files").rglob("*.json"):
            records.extend(json.loads(f.read_text())["annotations"])
        df = pd.DataFrame(records)
    assert len(df) == 91


@pytest.fixture(scope="module")
def parquet_file(tmp_path_factory, request):
    os.chdir(request.fspath.dirname)
    tmp_path = tmp_path_factory.mktemp("test_input_parquet")
    path = tmp_path / "input_test.pq"
    docs = edspdf.data.read_files(
        "file://" + os.path.abspath("../resources"),
        recursive=sys.version_info >= (3, 8),
    )
    docs.write_parquet(
        path,
        converter=lambda x: {
            "content": x["content"],
            "id": x["id"],
        },
    )
    os.chdir(request.config.invocation_dir)
    return path


@pytest.mark.parametrize("read_mode", ["parquet", "pandas", "iterable", "files"])
@pytest.mark.parametrize("num_cpu_workers", [1, 2])
@pytest.mark.parametrize("read_in_worker", [False, True])
def test_read_data(
    frozen_pipeline,
    tmp_path,
    parquet_file,
    change_test_dir,
    read_mode,
    num_cpu_workers,
    read_in_worker,
):
    if read_mode == "files":
        docs = edspdf.data.read_files(
            "file://" + os.path.abspath("../resources"),
            converter=lambda x: PDFDoc(id=x["id"], content=x["content"]),
            # read_in_worker=True,
        )
        if read_in_worker:
            pytest.skip()
    elif read_mode == "parquet":
        docs = edspdf.data.read_parquet(
            parquet_file,
            converter=lambda x: x["content"],
            read_in_worker=True,
        )
    elif read_mode == "pandas":
        if read_in_worker:
            pytest.skip()
        docs = edspdf.data.from_pandas(
            pd.read_parquet(parquet_file),
            converter=lambda x: x["content"],
        )
    else:
        if read_in_worker:
            pytest.skip()
        docs = edspdf.data.from_iterable(
            f.read_bytes() for f in Path("../resources").rglob("*.pdf")
        )
    docs = docs.map_pipeline(frozen_pipeline)
    docs = docs.set_processing(
        num_cpu_workers=num_cpu_workers,
        show_progress=True,
        batch_by="content_boxes",
        chunk_size=3,
        sort_chunks=True,
    )
    df = docs.to_pandas(converter=box_converter)
    assert len(df) == 91
