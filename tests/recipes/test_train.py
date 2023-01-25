from typing import Any, Callable, Generator

import datasets
from typer.testing import CliRunner

from edspdf import Pipeline, registry
from edspdf.models import Box, PDFDoc
from edspdf.recipes.train import app, train
from edspdf.utils.alignment import align_box_labels

runner = CliRunner()


@registry.adapter.register("segmentation-adapter")
def make_segmentation_adapter(
    path: str,
) -> Callable[[Pipeline], Generator[PDFDoc, Any, None]]:
    def adapt(model):
        for sample in datasets.load_from_disk(path):
            doc: PDFDoc = model.components.extractor(sample["content"])
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


def test_function(pdf, error_pdf, change_test_dir, dummy_dataset, tmp_path):
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe(
        "deep-classifier",
        name="classifier",
        config={
            "embedding": {
                "@factory": "box-embedding",
                "size": 72,
                "dropout_p": 0.1,
                "text_encoder": {
                    "@factory": "box-text-embedding",
                    "pooler": {
                        "@factory": "cnn-pooler",
                        "out_channels": 64,
                        "kernel_sizes": (3, 4, 5),
                    },
                },
                "layout_encoder": {
                    "@factory": "box-layout-embedding",
                    "n_positions": 64,
                    "x_mode": "learned",
                    "y_mode": "learned",
                    "w_mode": "learned",
                    "h_mode": "learned",
                },
                "contextualizer": {
                    "@factory": "box-transformer",
                    "input_size": 72,
                    "num_heads": 4,
                    "dropout_p": 0.1,
                    "activation": "gelu",
                    "init_resweight": 0.01,
                    "head_size": 16,
                    "attention_mode": ["c2c", "c2p", "p2c"],
                    "n_layers": 1,
                    "n_relative_positions": 64,
                },
            },
            "labels": [],
            "activation": "relu",
        },
    )
    print(model.cfg.to_str())

    train(
        model=model,
        train_data=make_segmentation_adapter(dummy_dataset),
        val_data=make_segmentation_adapter(dummy_dataset),
        max_steps=10,
        batch_size=4,
        validation_interval=4,
        output_dir=tmp_path,
    )

    list(model.pipe([pdf] * 2 + [error_pdf] * 2))
    output = model(PDFDoc(content=pdf))

    assert type(output) == PDFDoc


def test_script(change_test_dir, dummy_dataset):
    result = runner.invoke(
        app,
        [
            "--config",
            "config.cfg",
            "--batch_size",
            "4",
            "--validation_interval",
            "4",
            "--output-dir=null",
            "--components.aggregator.@factory=simple-aggregator",
            f"--train_data.path={dummy_dataset}",
            f"--val_data.path={dummy_dataset}",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Training model" in result.stdout
