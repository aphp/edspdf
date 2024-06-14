import itertools
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, List, Optional, Union

import datasets
import torch
from accelerate import Accelerator
from confit import Cli
from rich_logger import RichTablePrinter
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer.testing import CliRunner

import edspdf
from edspdf.pipeline import Pipeline
from edspdf.registry import registry
from edspdf.structures import Box, PDFDoc
from edspdf.utils.alignment import align_box_labels
from edspdf.utils.collections import flatten_dict
from edspdf.utils.optimization import LinearSchedule, ScheduledOptimizer
from edspdf.utils.random import set_seed

runner = CliRunner()


def score(golds, preds):
    return classification_report(
        [b.label for gold in golds for b in gold.text_boxes if b.text != ""],
        [b.label for pred in preds for b in pred.text_boxes if b.text != ""],
        output_dict=True,
        zero_division=0,
    )


@registry.adapter.register("segmentation-adapter")
def make_segmentation_adapter(
    path: str,
) -> Callable[[Pipeline], Generator[PDFDoc, Any, None]]:
    def adapt(model):
        for sample in datasets.load_from_disk(path):
            doc: PDFDoc = model.get_pipe("extractor")(sample["content"])
            doc.content_boxes = [
                box
                for page in doc.pages
                for box in align_box_labels(
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
                        if b["page"] == page.page_num
                    ],
                    dst_boxes=page.text_boxes,
                    pollution_label=None,
                )
                if box.text == "" or box.label is not None
            ]
            yield doc

    return adapt


app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="train", registry=registry)
def train(
    model: Pipeline,
    train_data: Callable[[Pipeline], Iterable[PDFDoc]],
    val_data: Union[Callable[[Pipeline], Iterable[PDFDoc]], float],
    batch_size: int = 2,
    max_steps: int = 2000,
    validation_interval: int = 100,
    lr: float = 8e-4,
    seed: int = 42,
    data_seed: int = 42,
    output_dir: Optional[Path] = Path("artifacts"),
):
    set_seed(seed)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        model_path = output_dir / "last-model"
        metrics_path = output_dir / "train_metrics.json"

    with RichTablePrinter(
        {
            "step": {},
            "(.*_)?loss": {
                "goal": "lower_is_better",
                "format": "{:.2e}",
                "goal_wait": 2,
            },
            "(?:.*/)?(.*)/accuracy": {
                "goal": "higher_is_better",
                "format": "{:.2%}",
                "goal_wait": 1,
                "name": r"\1/acc",
            },
            "lr": {"format": "{:.2e}"},
            "speed": {"format": "{:.2f}"},
        }
    ) as logger:
        with set_seed(data_seed):
            train_docs: List[PDFDoc] = list(train_data(model))
            val_docs: List[PDFDoc] = list(val_data(model))

        # Initialize pipeline with training documents
        model.post_init(iter(train_docs))

        # Preprocessing training data
        print("Preprocessing data")
        dataloader = DataLoader(
            list(model.preprocess_many(train_docs, supervision=True)),
            batch_size=batch_size,
            collate_fn=model.collate,
        )

        # Training loop
        print("Training", ", ".join([name for name, pipe in model.trainable_pipes()]))
        optimizer = ScheduledOptimizer(
            torch.optim.AdamW(
                [
                    {
                        "params": model.parameters(),
                        "lr": lr,
                        "schedules": LinearSchedule(
                            start_value=lr,
                            total_steps=max_steps,
                            warmup_rate=0.1,
                        ),
                    }
                ]
            )
        )

        accelerator = Accelerator(cpu=True)
        trained_pipes = torch.nn.ModuleDict(dict(model.trainable_pipes()))
        [dataloader, optimizer, trained_pipes] = accelerator.prepare(
            dataloader,
            optimizer,
            trained_pipes,
        )

        cumulated_losses = defaultdict(lambda: 0.0)

        iterator = itertools.chain.from_iterable(itertools.repeat(dataloader))
        all_metrics = []
        for step in tqdm(range(max_steps + 1), "Training model", leave=True):
            if (step % validation_interval) == 0:
                with model.select_pipes(enable=["classifier"]):
                    metrics = {
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        **cumulated_losses,
                        **flatten_dict(score(val_docs, model.pipe(deepcopy(val_docs)))),
                    }
                cumulated_losses = defaultdict(lambda: 0.0)
                all_metrics.append(metrics)
                logger.log_metrics(metrics)

                if output_dir is not None:
                    metrics_path.write_text(json.dumps(all_metrics, indent=2))
                    model.save(model_path)

                model.train()

            if step == max_steps:
                break

            optimizer.zero_grad()

            with model.cache():
                batch = next(iterator)
                loss = torch.zeros((), device=accelerator.device)
                for name, component in trained_pipes.items():
                    output = component.module_forward(batch[name])
                    if "loss" in output:
                        loss += output["loss"]
                        for key, value in output.items():
                            if key.endswith("loss"):
                                cumulated_losses[key] += float(value)

            accelerator.backward(loss)
            optimizer.step()


def test_function(pdf, error_pdf, change_test_dir, dummy_dataset, tmp_path):
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe(
        "box-transformer",
        name="embedding",
        config={
            "num_heads": 4,
            "dropout_p": 0.1,
            "activation": "gelu",
            "init_resweight": 0.01,
            "head_size": 16,
            "attention_mode": ["c2c", "c2p", "p2c"],
            "n_layers": 1,
            "n_relative_positions": 64,
            "embedding": {
                "@factory": "embedding-combiner",
                "dropout_p": 0.1,
                "text_encoder": {
                    "@factory": "sub-box-cnn-pooler",
                    "out_channels": 64,
                    "kernel_sizes": (3, 4, 5),
                    "embedding": {
                        "@factory": "simple-text-embedding",
                        "size": 72,
                    },
                },
                "layout_encoder": {
                    "@factory": "box-layout-embedding",
                    "n_positions": 64,
                    "x_mode": "learned",
                    "y_mode": "learned",
                    "w_mode": "learned",
                    "h_mode": "learned",
                    "size": 72,
                },
            },
        },
    )
    model.add_pipe(
        "trainable-classifier",
        name="classifier",
        config={
            "embedding": model.get_pipe("embedding"),
            "labels": [],
        },
    )
    print(model.config.to_str())

    data_adapter = make_segmentation_adapter(dummy_dataset)

    train(
        model=model,
        train_data=data_adapter,
        val_data=data_adapter,
        max_steps=20,
        batch_size=4,
        validation_interval=4,
        output_dir=tmp_path,
        lr=0.001,
    )

    docs = list(data_adapter(model))

    model = edspdf.load(tmp_path / "last-model")

    list(model.pipe([pdf] * 2 + [error_pdf] * 2))
    output = model(PDFDoc(content=pdf))

    with model.select_pipes(enable=["classifier"]):
        assert score(docs, model.pipe(deepcopy(docs)))["accuracy"] > 0.5

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


def test_function_huggingface(pdf, error_pdf, change_test_dir, dummy_dataset, tmp_path):
    set_seed(42)
    model = Pipeline()
    model.add_pipe("pdfminer-extractor", name="extractor")
    model.add_pipe(
        "huggingface-embedding",
        name="embedding",
        config={
            "model": "hf-tiny-model-private/tiny-random-LayoutLMv3Model",
            "window": 64,
            "stride": 32,
            "use_image": False,
        },
    )
    model.add_pipe(
        "trainable-classifier",
        name="classifier",
        config={
            "embedding": model.get_pipe("embedding"),
            "labels": [],
        },
    )
    trf = model.get_pipe("embedding")
    trf.hf_model.encoder.layer = trf.hf_model.encoder.layer[:1]

    data_adapter = make_segmentation_adapter(dummy_dataset)

    train(
        model=model,
        train_data=data_adapter,
        val_data=data_adapter,
        max_steps=2,
        batch_size=1,
        validation_interval=1,
        output_dir=tmp_path,
        lr=0.001,
    )

    docs = list(data_adapter(model))

    model = edspdf.load(tmp_path / "last-model")
    assert not model.get_pipe("embedding").training

    weird_doc: PDFDoc = model.get_pipe("extractor")(pdf)
    for line in weird_doc.text_boxes:
        line.text = ""
    list(model.pipe([]))
    with model.select_pipes(disable=["extractor"]):
        list(model.pipe([weird_doc] * 4))
    list(model.pipe([error_pdf] * 4))
    list(model.pipe([]))
    list(model.pipe([pdf] * 2 + [error_pdf] * 2))
    output = model(PDFDoc(content=pdf))

    with model.select_pipes(enable=["classifier"]):
        assert score(docs, model.pipe(deepcopy(docs)))["accuracy"] > 0.5

    assert type(output) == PDFDoc
