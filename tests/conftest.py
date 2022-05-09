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
def extractor():
    return PdfMinerExtractor(style=True)


@fixture
def date_text():
    text = (
        "Examen du 24/06/2014, 25 JUIN 2014, "
        "imprimé le 2-9-2014 puis signé le 28.10.2014"
    )
    return text
