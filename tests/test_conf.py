from thinc.config import Config

from edspdf import registry
from edspdf.reading import PdfReader

configuration = """
[reader]
@readers = "pdf-reader.v1"

[reader.aggregator]
@aggregators = "simple.v1"
new_line_threshold = 0.2
new_paragraph_threshold = 1.2

[reader.extractor]
@extractors = "line-extractor.v1"
style = true
laparams = { "@params": "laparams.v1" }

[reader.classifier]
@classifiers = "dummy.v1"

[reader.transform]
@transforms = "chain.v1"

[reader.transform.*.dates]
@transforms = "dates.v1"

[reader.transform.*.telephone]
@transforms = "telephone.v1"

[reader.transform.*.dimensions]
@transforms = "dimensions.v1"

"""


def test_configuration():
    cfg = Config().from_str(configuration)
    resolved = registry.resolve(cfg)

    assert isinstance(resolved["reader"], PdfReader)
