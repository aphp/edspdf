from edspdf.pipes.classifiers.mask import simple_mask_classifier_factory
from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor
from edspdf.visualization import compare_results, merge_boxes, show_annotations


def test_pipeline(pdf):
    extractor = PdfMinerExtractor()
    classifier = simple_mask_classifier_factory(
        x0=0.1, y0=0.4, x1=0.5, y1=0.9, threshold=0.1
    )

    doc = extractor(pdf)
    doc = classifier(doc)

    merged = merge_boxes(doc.lines)

    assert len(show_annotations(pdf, merged)) == 1
    assert len(compare_results(pdf, doc.lines, merged)) == 1
