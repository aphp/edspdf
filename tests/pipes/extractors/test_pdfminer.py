import pytest
from blocks_ground_truth import blank_blocks, pdf_blocks, styles_blocks
from pdfminer.pdfparser import PDFSyntaxError

from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor


def test_pdfminer(pdf, styles_pdf, blank_pdf):
    extractor = PdfMinerExtractor(extract_style=False)

    pytest.nested_approx(extractor(pdf).text_boxes, pdf_blocks, abs=5e-2)
    pytest.nested_approx(extractor(styles_pdf).text_boxes, styles_blocks, abs=5e-2)
    pytest.nested_approx(extractor(blank_pdf).text_boxes, blank_blocks, abs=5e-2)


def test_pdfminer_image(pdf, styles_pdf, blank_pdf):
    extractor = PdfMinerExtractor(extract_style=False, render_pages=True)

    assert extractor(pdf).pages[0].image.shape == (2339, 1654, 3)
    assert extractor(styles_pdf).pages[0].image.shape == (2200, 1700, 3)
    assert extractor(blank_pdf).pages[0].image.shape == (2339, 1654, 3)


def test_pdfminer_error(error_pdf):
    extractor = PdfMinerExtractor(extract_style=False, raise_on_error=True)

    with pytest.raises(PDFSyntaxError):
        extractor(error_pdf)

    extractor.raise_on_error = False
    result = extractor(error_pdf)
    assert len(result.text_boxes) == 0
    assert result.error is True
