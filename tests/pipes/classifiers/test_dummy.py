from edspdf.pipes.classifiers.dummy import DummyClassifier


def test_dummy(single_page_doc):
    classifier = DummyClassifier(label="body")

    single_page_doc = classifier(single_page_doc)

    p1, p2, p3 = [b.label for b in single_page_doc.text_boxes]

    assert p1 == "body"
    assert p2 == "body"
    assert p3 == "body"
