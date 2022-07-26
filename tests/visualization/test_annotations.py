from edspdf.classifiers.mask import simple_mask_classifier_factory
from edspdf.extractors.pdfminer import PdfMinerExtractor
from edspdf.visualization import compare_results, merge_lines, show_annotations


def test_pipeline(pdf):

    extractor = PdfMinerExtractor()
    classifier = simple_mask_classifier_factory(
        x0=0.1, y0=0.4, x1=0.5, y1=0.9, threshold=0.1
    )

    df = extractor(pdf)
    df["label"] = classifier(df)

    merged = merge_lines(df)

    show_annotations(pdf, merged)
    compare_results(pdf, merged, merged)
