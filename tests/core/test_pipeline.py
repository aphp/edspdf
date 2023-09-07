import copy
from itertools import chain
from pathlib import Path

import datasets
import pytest
import torch
from confit import Config
from confit.errors import ConfitValidationError
from confit.registry import validate_arguments
from sklearn.metrics import classification_report

import edspdf
import edspdf.accelerators.multiprocessing
from edspdf import TrainablePipe
from edspdf.pipeline import Pipeline
from edspdf.pipes.aggregators.simple import SimpleAggregator
from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor
from edspdf.registry import registry
from edspdf.structures import Box, PDFDoc
from edspdf.utils.alignment import align_box_labels


class CustomClass:
    pass


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
    assert model.has_pipe("pdfminer-extractor")

    model.add_pipe("simple-aggregator", name="aggregator")
    assert "aggregator" in model.pipe_names
    assert model.has_pipe("aggregator")

    with pytest.raises(ValueError):
        model.get_pipe("missing-pipe")


def test_add_pipe_component():
    model = Pipeline()
    model.add_pipe(PdfMinerExtractor(pipeline=model, name="pdfminer-extractor"))
    assert "pdfminer-extractor" in model.pipe_names
    assert model.has_pipe("pdfminer-extractor")

    model.add_pipe(SimpleAggregator(pipeline=model, name="aggregator"))
    assert "aggregator" in model.pipe_names
    assert model.has_pipe("aggregator")

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
    def score(golds, preds):
        return classification_report(
            [b.label for gold in golds for b in gold.text_boxes if b.text != ""],
            [b.label for pred in preds for b in pred.text_boxes if b.text != ""],
            output_dict=True,
            zero_division=0,
        )

    assert type(pipeline(pdf)) == PDFDoc
    assert len(list(pipeline.pipe([pdf] * 4))) == 4

    data = list(make_segmentation_adapter(dummy_dataset)(pipeline))
    with pipeline.select_pipes(enable=["classifier"]):
        results = score(data, pipeline.pipe(copy.deepcopy(data)))
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
    with pipeline.select_pipes(enable="extractor"):
        assert all(box.label is None for box in pipeline(pdf).content_boxes)
    with pytest.raises(ValueError):
        with pipeline.select_pipes(disable="aggregator"):
            pass
    with pipeline.select_pipes(disable="simple-aggregator"):
        assert pipeline(pdf).aggregated_texts == {}


def test_different_names(pipeline: Pipeline):
    pipeline = Pipeline()

    extractor = PdfMinerExtractor(pipeline=pipeline, name="custom_name")

    with pytest.raises(ValueError) as exc_info:
        pipeline.add_pipe(extractor, name="extractor")

    assert "The provided name does not match the name of the component." in str(
        exc_info.value
    )


fail_config = """
[train]
model = ${pipeline}
max_steps = 20
lr = 8e-4
seed = 43

[train.train_data]
@adapter = segmentation-adapter

[train.val_data]
@adapter = segmentation-adapter

[pipeline]
pipeline = ["extractor", "embedding", "classifier"]
disabled = []

[components]

[components.extractor]
@factory = "pdfminer-extractor"

[components.classifier]
@factory = "trainable-classifier"
labels = []
embedding = ${components.embedding}

[components.embedding]
@factory = "box-layout-embedding"
n_positions = 64
x_mode = "learned"
y_mode = "learned"
w_mode = "learned"
h_mode = "hello"
"""


def test_config_validation_error():
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")

    with pytest.raises(ConfitValidationError) as e:
        Pipeline.from_config(Config.from_str(fail_config))

    assert str(e.value) == (
        "2 validation errors for edspdf.pipeline.Pipeline()\n"
        "-> components.components.classifier.embedding.size\n"
        "   field required\n"
        "-> components.components.classifier.embedding.h_mode\n"
        "   unexpected value; permitted: 'sin', 'learned', got 'hello' (str)"
    )


def test_add_pipe_validation_error():
    model = Pipeline()
    with pytest.raises(ConfitValidationError) as e:
        model.add_pipe("pdfminer-extractor", name="extractor", config={"foo": "bar"})

    assert str(e.value) == (
        "1 validation error for edspdf.pipes.extractors.pdfminer.PdfMinerExtractor()\n"
        "-> extractor.foo\n"
        "   unexpected keyword argument"
    )


def test_multiprocessing_accelerator(frozen_pipeline, pdf, letter_pdf):
    edspdf.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    docs = list(
        frozen_pipeline.pipe(
            [pdf, letter_pdf] * 20,
            accelerator="multiprocessing",
            batch_size=2,
        )
    )
    assert len(docs) == 40


def error_pipe(doc: PDFDoc):
    if doc.id == "pdf-3":
        raise ValueError("error")
    return doc


def test_multiprocessing_gpu_stub(frozen_pipeline, pdf, letter_pdf):
    edspdf.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    accelerator = edspdf.accelerators.multiprocessing.MultiprocessingAccelerator(
        batch_size=2,
        num_gpu_workers=1,
        num_cpu_workers=1,
        gpu_worker_devices=["cpu"],
    )
    list(
        frozen_pipeline.pipe(
            chain.from_iterable(
                [
                    {"content": pdf},
                    {"content": letter_pdf},
                ]
                for i in range(5)
            ),
            accelerator=accelerator,
            to_doc="content",
            from_doc={"text": "aggregated_texts"},
        )
    )


def test_multiprocessing_rb_error(pipeline, pdf, letter_pdf):
    edspdf.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    pipeline.add_pipe(error_pipe, name="error", after="extractor")
    with pytest.raises(ValueError):
        list(
            pipeline.pipe(
                chain.from_iterable(
                    [
                        {"content": pdf, "id": f"pdf-{i}"},
                        {"content": letter_pdf, "id": f"letter-{i}"},
                    ]
                    for i in range(5)
                ),
                accelerator="multiprocessing",
                batch_size=2,
                to_doc={"content_field": "content", "id_field": "id"},
            )
        )


class DeepLearningError(TrainablePipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, doc):
        return {"num_boxes": len(doc.content_boxes), "doc_id": doc.id}

    def collate(self, batch, device):
        return {
            "num_boxes": torch.tensor(batch["num_boxes"], device=device),
            "doc_id": batch["doc_id"],
        }

    def forward(self, batch):
        if "pdf-1" in batch["doc_id"]:
            raise RuntimeError("Deep learning error")
        return {}


def test_multiprocessing_ml_error(pipeline, pdf, letter_pdf):
    edspdf.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    pipeline.add_pipe(
        DeepLearningError(
            pipeline=pipeline,
            name="error",
        ),
        after="extractor",
    )
    accelerator = edspdf.accelerators.multiprocessing.MultiprocessingAccelerator(
        batch_size=2,
        num_gpu_workers=1,
        num_cpu_workers=1,
        gpu_worker_devices=["cpu"],
    )
    with pytest.raises(RuntimeError) as e:
        list(
            pipeline.pipe(
                chain.from_iterable(
                    [
                        {"content": pdf, "id": f"pdf-{i}"},
                        {"content": letter_pdf, "id": f"letter-{i}"},
                    ]
                    for i in range(5)
                ),
                accelerator=accelerator,
                to_doc={"content_field": "content", "id_field": "id"},
            )
        )
    assert "Deep learning error" in str(e.value)
