from pathlib import Path

import datasets
import pytest
from confit import Config
from confit.registry import validate_arguments

import edspdf
from edspdf.pipeline import Pipeline
from edspdf.pipes.aggregators.simple import SimpleAggregator
from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor
from edspdf.registry import registry
from edspdf.structures import Box, PDFDoc
from edspdf.utils.alignment import align_box_labels


class CustomClass:
    pass


@pytest.fixture(scope="session")
def frozen_pipeline():
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe(
        "trainable-classifier",
        name="classifier",
        config=dict(
            embedding={
                "@factory": "box-layout-embedding",
                "n_positions": 32,
                "size": "48",
            },
            labels=["first", "second"],
        ),
    )
    model.add_pipe("simple-aggregator")
    model.post_init([])
    return model


@pytest.fixture()
def pipeline():
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe(
        "trainable-classifier",
        name="classifier",
        config=dict(
            embedding={
                "@factory": "box-layout-embedding",
                "n_positions": 32,
                "size": "48",
            },
            labels=["first", "second"],
        ),
    )
    model.add_pipe("simple-aggregator")
    model.post_init([])
    return model


def test_add_pipe_factory():
    model = Pipeline()
    model.add_pipe("pdfminer-extractor")
    assert "pdfminer-extractor" in model.pipe_names

    model.add_pipe("simple-aggregator", name="aggregator")
    assert "aggregator" in model.pipe_names


def test_add_pipe_component():
    model = Pipeline()
    model.add_pipe(PdfMinerExtractor(pipeline=model, name="pdfminer-extractor"))
    assert "pdfminer-extractor" in model.pipe_names

    model.add_pipe(SimpleAggregator(pipeline=model, name="aggregator"))
    assert "aggregator" in model.pipe_names

    with pytest.raises(ValueError):
        model.add_pipe(
            SimpleAggregator(pipeline=model, name="aggregator"),
            config={"label_map": {"table": "body"}},
        )

    with pytest.raises(ValueError):
        model.add_pipe(CustomClass())


def test_sequence(frozen_pipeline: Pipeline):
    assert len(frozen_pipeline.pipeline) == 3
    assert list(frozen_pipeline.pipeline) == [
        ("extractor", frozen_pipeline.get_pipe("extractor")),
        ("classifier", frozen_pipeline.get_pipe("classifier")),
        ("simple-aggregator", frozen_pipeline.get_pipe("simple-aggregator")),
    ]
    assert list(frozen_pipeline.trainable_pipes()) == [
        ("classifier", frozen_pipeline.get_pipe("classifier")),
    ]


def test_serialization(frozen_pipeline: Pipeline):
    assert (
        frozen_pipeline.config.to_str()
        == """\
[pipeline]
pipeline = ["extractor", "classifier", "simple-aggregator"]
disabled = []
components = ${components}

[components]

[components.extractor]
@factory = "pdfminer-extractor"

[components.classifier]
@factory = "trainable-classifier"
labels = ["first", "second"]

[components.classifier.embedding]
@factory = "box-layout-embedding"
n_positions = 32
size = "48"

[components.simple-aggregator]
@factory = "simple-aggregator"

"""
    )  # noqa: E501


config_str = """
[model]
pipeline = ['pdfminer-extractor', 'trainable-classifier', 'simple-aggregator']
components = ${components}

[components]

[components.'pdfminer-extractor']
@factory = 'pdfminer-extractor'

[components.'trainable-classifier']
@factory = 'trainable-classifier'
labels = ['first', 'second']

[components.'trainable-classifier'.embedding]
@factory = 'box-layout-embedding'
size = 96
n_positions = 32

[components.'simple-aggregator']
@factory = 'simple-aggregator'"""


def test_validate_config():
    @validate_arguments
    def function(model: Pipeline):
        print(model.pipe_names)
        assert len(model.pipe_names) == 3

    function(Config.from_str(config_str).resolve(registry=registry)["model"])


def test_load_config(change_test_dir):
    pipeline = edspdf.load("config.cfg")
    assert pipeline.pipe_names == ["extractor", "classifier"]


def test_torch_module(frozen_pipeline: Pipeline):
    with frozen_pipeline.train(True):
        for name, component in frozen_pipeline.trainable_pipes():
            assert component.training is True

    with frozen_pipeline.train(False):
        for name, component in frozen_pipeline.trainable_pipes():
            assert component.training is False

    frozen_pipeline.to("cpu")


def make_segmentation_adapter(path: str):
    def adapt(model):
        for sample in datasets.load_from_disk(path):
            doc = model.get_pipe("extractor")(sample["content"])
            doc.content_boxes = [
                b
                for page_num in sorted(set(b.page_num for b in doc.lines))
                for b in align_box_labels(
                    src_boxes=[
                        Box(
                            doc=doc,
                            page_num=b["page"],
                            x0=b["x0"],
                            x1=b["x1"],
                            y0=b["y0"],
                            y1=b["y1"],
                            label=b["label"]
                            if b["label"] not in ("section_title", "table")
                            else "body",
                        )
                        for b in sample["bboxes"]
                        if b["page"] == page_num
                    ],
                    dst_boxes=doc.lines,
                    pollution_label=None,
                )
                if b.text == "" or b.label is not None
            ]
            yield doc

    return adapt


def test_pipeline_on_data(pipeline: Pipeline, dummy_dataset: str, pdf: bytes):
    assert type(pipeline(pdf)) == PDFDoc
    assert len(list(pipeline.pipe([pdf] * 4))) == 4

    data = list(make_segmentation_adapter(dummy_dataset)(pipeline))
    results = pipeline.score(data)
    assert isinstance(results, dict)


def test_cache(pipeline: Pipeline, dummy_dataset: Path, pdf: bytes):
    pipeline(pdf)

    with pipeline.cache():
        pipeline(pdf)
        assert len(pipeline._cache) > 0

    assert pipeline._cache is None


def test_select_pipes(pipeline: Pipeline, pdf: bytes):
    with pipeline.select_pipes(enable=["extractor", "classifier"]):
        assert pipeline(pdf).aggregated_texts == {}


def test_different_names(pipeline: Pipeline):
    pipeline = Pipeline()

    extractor = PdfMinerExtractor(pipeline=pipeline, name="custom_name")

    with pytest.raises(ValueError) as exc_info:
        pipeline.add_pipe(extractor, name="extractor")

    assert "The provided name does not match the name of the component." in str(
        exc_info.value
    )
