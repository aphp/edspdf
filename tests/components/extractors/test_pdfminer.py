import pytest
from blocks_ground_truth import blank_blocks, pdf_blocks, styles_blocks
from pdfminer.pdfparser import PDFSyntaxError

from edspdf.components.extractors.pdfminer import PdfMinerExtractor


def test_pdfminer(pdf, styles_pdf, blank_pdf):
    extractor = PdfMinerExtractor(extract_style=False)

    pytest.nested_approx(extractor(pdf).lines, pdf_blocks, abs=5e-2)
    pytest.nested_approx(extractor(styles_pdf).lines, styles_blocks, abs=5e-2)
    pytest.nested_approx(extractor(blank_pdf).lines, blank_blocks, abs=5e-2)


def test_pdfminer_error(error_pdf):
    extractor = PdfMinerExtractor(extract_style=False, raise_on_error=True)

    with pytest.raises(PDFSyntaxError):
        extractor(error_pdf)

    extractor.raise_on_error = False
    result = extractor(error_pdf)
    assert len(result.lines) == 0
    assert result.error is True
