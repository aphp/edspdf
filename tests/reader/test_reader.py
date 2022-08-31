from thinc.config import Config

from edspdf import registry

configuration = """
[reader]
@readers = "pdf-reader.v1"

[reader.aggregator]
@aggregators = "simple.v1"
new_line_threshold = 0.2
new_paragraph_threshold = 1.5
label_map = {"title": "body"}

[reader.extractor]
@extractors = "pdfminer.v1"

[reader.classifier]
@classifiers = "custom_masks.v1"

[reader.classifier.title]
label = "title"
x0 = 0.2
y0 = 0.3
x1 = 1.0
y1 = 0.4
threshold = 0.9

[reader.classifier.body]
label = "body"
x0 = 0.2
y0 = 0.4
x1 = 1.0
y1 = 1.0
threshold = 0.9

[reader.transform]
@transforms = "chain.v1"

[reader.transform.*.dates]
@transforms = "dates.v1"

[reader.transform.*.telephone]
@transforms = "telephone.v1"

[reader.transform.*.rescale]
@transforms = "rescale.v1"

[reader.transform.*.dimensions]
@transforms = "dimensions.v1"
"""


def test_reader(pdf, blank_pdf, letter_pdf):
    cfg = Config().from_str(configuration)
    resolved = registry.resolve(cfg)

    reader = resolved["reader"]

    reader(pdf, orbis=True)
    reader(blank_pdf, orbis=True)
    assert reader(letter_pdf)["body"] == (
        "Cher Pr ABC, Cher DEF,\n"
        "\n"
        "Nous souhaitons remercier le CSE pour son avis favorable quant à l’accès aux "
        "données de\n"
        "l’Entrepôt de Données de Santé du projet n° XXXX.\n"
        "\n"
        "Nous avons bien pris connaissance des conditions requises pour cet avis "
        "favorable, c’est\n"
        "pourquoi nous nous engageons par la présente à :\n"
        "\n"
        "• Informer individuellement les patients concernés par la recherche, admis à "
        "l'AP-HP\n"
        "avant juillet 2017, sortis vivants, et non réadmis depuis.\n"
        "\n"
        "• Effectuer une demande d'autorisation à la CNIL en cas d'appariement avec "
        "d’autres\n"
        "cohortes.\n"
        "\n"
        "Bien cordialement,\n"
        "\n"
        "Pr XXXX Pr YYYY\n"
        "\n"
    )
