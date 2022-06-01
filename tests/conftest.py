from pathlib import Path

from pytest import fixture

from edspdf.extraction.pdfminer import PdfMinerExtractor

TEST_DIR = Path(__file__).parent


class DummyClassifier:
    def predict(self, X):
        return ["body" for _ in range(len(X))]


@fixture
def pdf():
    path = TEST_DIR / "resources" / "test.pdf"
    return path.read_bytes()


@fixture
def blank_pdf():
    path = TEST_DIR / "resources" / "blank.pdf"
    return path.read_bytes()
