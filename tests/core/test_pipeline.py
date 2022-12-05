from pathlib import Path

import datasets
import pytest

from edspdf import Config, Pipeline, TrainableComponent
from edspdf.cli import validate_arguments
from edspdf.components import PdfMinerExtractor, SimpleAggregator
from edspdf.models import Box
from edspdf.utils.alignment import align_box_labels


class CustomClass:
    pass


@pytest.fixture(scope="session")
def frozen_pipeline():
    model = Pipeline()
    model.add_pipe("pdfminer-extractor")
    model.add_pipe(
        "deep-classifier",
        config=dict(
            embedding={
                "@factory": "text-box-embedding",
                "size": "96",
                "box_encoder": {
                    "n_positions": 32,
                },
            },
            labels=["first", "second"],
            do_harmonize=False,
        ),
    )
    model.add_pipe("simple-aggregator")
    return model


@pytest.fixture()
def pipeline(empty_dataset):
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe(
        "deep-classifier",
        name="classifier",
        config=dict(
            embedding={
                "@factory": "text-box-embedding",
                "size": "96",
                "box_encoder": {"n_positions": 32},
            },
            labels=["first", "second"],
            do_harmonize=False,
        ),
    )
    model.add_pipe("simple-aggregator")
    model.initialize([])
    return model


def test_add_pipe_factory():
    model = Pipeline()
    model.add_pipe("pdfminer-extractor")
    assert "pdfminer-extractor" in model.components

    model.add_pipe("simple-aggregator", name="aggregator")
    assert "aggregator" in model.components


def test_add_pipe_component():
    model = Pipeline()
    model.add_pipe(PdfMinerExtractor())
    assert "pdfminer-extractor" in model.components

    model.add_pipe(SimpleAggregator(), name="aggregator")
    assert "aggregator" in model.components

    with pytest.raises(TypeError):
        model.add_pipe(SimpleAggregator(), config={"label_map": {"table": "body"}})

    with pytest.raises(TypeError):
        model.add_pipe(CustomClass())


def test_sequence(frozen_pipeline: Pipeline):
    assert len(frozen_pipeline) == 3
    assert list(frozen_pipeline) == [
        frozen_pipeline.components["pdfminer-extractor"],
        frozen_pipeline.components["deep-classifier"],
        frozen_pipeline.components["simple-aggregator"],
    ]
    assert frozen_pipeline.trainable_components == [
        frozen_pipeline.components["deep-classifier"],
    ]


def test_serialization(frozen_pipeline: Pipeline):
    print(Config(model=frozen_pipeline).to_str())
    assert (
        repr(frozen_pipeline)
        == """\
Pipeline(
  (pdfminer-extractor): PdfMinerExtractor()
  (deep-classifier): DeepClassifier(
    (label_vocabulary): Vocabulary(n=3)
    (embedding): TextBoxEmbedding(
      (box_encoder): BoxEmbedding(
        (box_preprocessor): BoxPreprocessor()
        (x_embedding): SinusoidalEmbedding(32, 16)
        (y_embedding): SinusoidalEmbedding(32, 16)
        (w_embedding): SinusoidalEmbedding(32, 16)
        (h_embedding): SinusoidalEmbedding(32, 16)
      )
      (text_encoder): TextEmbedding(
        (shape_voc): Vocabulary(n=1)
        (prefix_voc): Vocabulary(n=1)
        (suffix_voc): Vocabulary(n=1)
        (norm_voc): Vocabulary(n=1)
        (pooler): CnnPooler(
          (convolutions): ModuleList(
            (0): Conv1d(96, 96, kernel_size=(3,), stride=(1,))
            (1): Conv1d(96, 96, kernel_size=(4,), stride=(1,))
            (2): Conv1d(96, 96, kernel_size=(5,), stride=(1,))
          )
          (linear): Linear(in_features=288, out_features=96, bias=True)
        )
      )
      (dropout): Dropout(p=0.2, inplace=False)
    )
    (linear): Linear(in_features=96, out_features=96, bias=True)
    (dropout): Dropout(p=0.15, inplace=False)
  )
  (simple-aggregator): SimpleAggregator()
)"""
    )

    assert (
        Config(model=frozen_pipeline).to_str()
        == """\
[model]
components = ["pdfminer-extractor","deep-classifier","simple-aggregator"]
components_config = ${components}

[components]

[components.pdfminer-extractor]
@factory = "pdfminer-extractor"

[components.deep-classifier]
@factory = "deep-classifier"
labels = ["first","second"]
do_harmonize = false

[components.deep-classifier.embedding]
@factory = "text-box-embedding"
size = 96

[components.deep-classifier.embedding.box_encoder]
n_positions = 32

[components.simple-aggregator]
@factory = "simple-aggregator"

"""
    )  # noqa: E501


def test_load_config():
    config_str = """[model]
components = ['pdfminer-extractor', \
'deep-classifier', 'simple-aggregator']
components_config = ${components}

[components]

[components.'pdfminer-extractor']
@factory = 'pdfminer-extractor'

[components.'deep-classifier']
@factory = 'deep-classifier'
labels = ['first', 'second']

[components.'deep-classifier'.embedding]
@factory = 'text-box-embedding'
size = 96
box_encoder = {"n_positions": 32}

[components.'simple-aggregator']
@factory = 'simple-aggregator'"""  # noqa: E501

    @validate_arguments
    def function(model: Pipeline):
        assert len(model) == 3

    function(Config().from_str(config_str)["model"])


def test_torch_module(frozen_pipeline: Pipeline):
    frozen_pipeline.train(True)
    for component in frozen_pipeline.trainable_components:
        assert component.training is True

    frozen_pipeline.train(False)
    for component in frozen_pipeline.trainable_components:
        assert component.training is False


def make_segmentation_adapter(path: str):
    def adapt(model):
        for sample in datasets.load_from_disk(path):
            doc = model.components.extractor(sample["content"])
            doc.lines = [
                b
                for page in sorted(set(b.page for b in doc.lines))
                for b in align_box_labels(
                    src_boxes=[
                        Box(
                            page=b["page"],
                            x0=b["x0"],
                            x1=b["x1"],
                            y0=b["y0"],
                            y1=b["y1"],
                            label=b["label"]
                            if b["label"] not in ("section_title", "table")
                            else "body",
                        )
                        for b in sample["bboxes"]
                        if b["page"] == page
                    ],
                    dst_boxes=doc.lines,
                    pollution_label=None,
                )
                if b.text == "" or b.label is not None
            ]
            yield doc

    return adapt


def test_pipeline_on_data(pipeline: Pipeline, dummy_dataset: str, pdf: bytes):
    assert type(pipeline(pdf)) == dict
    assert len(list(pipeline.pipe([pdf] * 4))) == 4

    data = list(make_segmentation_adapter(dummy_dataset)(pipeline))
    results = pipeline.score(data)
    assert isinstance(results, dict)


def test_cache(pipeline: Pipeline, dummy_dataset: Path, pdf: bytes):
    pipeline.reset_cache()

    pipeline(pdf)
    embedding: TrainableComponent = pipeline.components.classifier.embedding
    assert len(embedding._collate_cache) == 1, "collate"
    assert len(embedding._preprocess_cache) == 1, "preprocess"
    assert len(embedding._forward_cache) == 1, "forward"

    pipeline.reset_cache()

    assert len(embedding._collate_cache) == 0, "collate"
    assert len(embedding._preprocess_cache) == 0, "preprocess"
    assert len(embedding._forward_cache) == 0, "forward"

    with pipeline.no_cache():
        pipeline(pdf)
        embedding: TrainableComponent = pipeline.components.classifier.embedding
        assert len(embedding._collate_cache) == 0, "collate"
        assert len(embedding._preprocess_cache) == 0, "preprocess"
        assert len(embedding._forward_cache) == 0, "forward"
