import os
from pathlib import Path

import pytest
from datasets import Dataset
from pytest import fixture
from utils import nested_approx

from edspdf import Pipeline
from edspdf.utils.collections import ld_to_dl

pytest.nested_approx = nested_approx

TEST_DIR = Path(__file__).parent


@pytest.fixture
def change_test_dir(request):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_dir)


@fixture(scope="session")
def pdf():
    path = TEST_DIR / "resources" / "test.pdf"
    return path.read_bytes()


@fixture(scope="session")
def blank_pdf():
    path = TEST_DIR / "resources" / "blank.pdf"
    return path.read_bytes()


@fixture(scope="session")
def styles_pdf():
    path = TEST_DIR / "resources" / "styles.pdf"
    return path.read_bytes()


@fixture(scope="session")
def letter_pdf():
    path = TEST_DIR / "resources" / "letter.pdf"
    return path.read_bytes()


@fixture(scope="session")
def error_pdf():
    path = TEST_DIR / "resources" / "error.pdf"
    return path.read_bytes()


@fixture(scope="session")
def dummy_dataset(tmpdir_factory, pdf):
    tmp_path = tmpdir_factory.mktemp("datasets")
    dataset_path = str(tmp_path / "pdf-dataset.hf")

    ds = Dataset.from_dict(
        ld_to_dl(
            [
                {
                    "id": str(i),
                    "content": pdf,
                    "bboxes": [
                        {
                            "page": 0,
                            "x0": 0.1,
                            "y0": 0.1,
                            "x1": 0.9,
                            "y1": 0.5,
                            "label": "first",
                            "page_width": 20,
                            "page_height": 30,
                        },
                        {
                            "page": 0,
                            "x0": 0.1,
                            "y0": 0.6,
                            "x1": 0.9,
                            "y1": 0.9,
                            "label": "second",
                            "page_width": 20,
                            "page_height": 30,
                        },
                    ],  # top half part of the page with margin
                }
                for i in range(8)
            ]
        )
    )
    ds.save_to_disk(dataset_path)
    return dataset_path


@pytest.fixture(scope="session")
def frozen_pipeline():
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe(
        "trainable-classifier",
        name="classifier",
        config=dict(
            embedding={
                "@factory": "box-layout-embedding",
                "n_positions": 32,
                "size": "48",
            },
            labels=["first", "second"],
        ),
    )
    model.add_pipe("simple-aggregator")
    model.post_init([])
    return model
