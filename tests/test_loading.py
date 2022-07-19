from pathlib import Path
from pytest import fixture
from edspdf import load, from_str


CONFIG = """
[reader]
@readers = "pdf-reader.v1"

[reader.extractor]
@extractors = "pdfminer.v1"

[reader.classifier]
@classifiers = "mask.v1"
x0 = 0.1
x1 = 0.9
y0 = 0.4
y1 = 0.9
threshold = 0.1

[reader.aggregator]
@aggregators = "simple.v1"
"""


@fixture
def config(tmp_path) -> Path:
    path = tmp_path / "config.cfg"

    path.write_text(CONFIG)

    return path


def test_load(config: Path, pdf: bytes):
    reader = load(config)
    reader(pdf)


def test_load_from_str(pdf: bytes):
    reader = from_str(CONFIG)
    reader(pdf)
