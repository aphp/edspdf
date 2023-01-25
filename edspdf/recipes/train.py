import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import torch
from accelerate import Accelerator
from rich_logger import RichTablePrinter
from torch.utils.data import DataLoader
from tqdm import tqdm

from edspdf import Cli, Pipeline
from edspdf.models import PDFDoc
from edspdf.utils.collections import dedup, flatten_dict
from edspdf.utils.optimization import LinearSchedule, ScheduledOptimizer
from edspdf.utils.random import set_seed

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="train")
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
    return_model: bool = False,
):
    # for key, value in locals().items():
    #     print("{}: {}".format(str(key).ljust(20), value))

    set_seed(seed)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        model_path = output_dir / "model.pt"
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
            if isinstance(val_data, float):
                offset = int(len(train_docs) * (1 - val_data))
                random.shuffle(train_docs)
                train_docs, val_docs = train_docs[:offset], train_docs[offset:]
            else:
                val_docs: List[PDFDoc] = list(val_data(model))

        # Initialize pipeline with training documents
        model.initialize(train_docs)

        # Preprocessing training data
        print("Preprocessing data")
        dataloader = DataLoader(
            list(model.preprocess_many(train_docs, supervision=True)),
            batch_size=batch_size,
            collate_fn=model.collate,
        )

        trained_pipes = model.trainable_components

        # Training loop
        print("Training", ", ".join([c.name for c in trained_pipes]))
        optimizer = ScheduledOptimizer(
            torch.optim.AdamW(
                [
                    {
                        "params": dedup(
                            [param for c in trained_pipes for param in c.parameters()]
                        ),
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

        accelerator = Accelerator()
        [dataloader, optimizer, *trained_pipes] = accelerator.prepare(
            dataloader,
            optimizer,
            *trained_pipes,
        )

        cumulated_losses = defaultdict(lambda: 0.0)

        iterator = itertools.chain.from_iterable(itertools.repeat(dataloader))
        all_metrics = []
        for step in tqdm(range(max_steps + 1), "Training model", leave=True):
            if (step % validation_interval) == 0:
                metrics = {
                    "step": step,
                    "lr": optimizer.param_groups[0]["lr"],
                    **cumulated_losses,
                    **flatten_dict(model.score(val_docs)),
                }
                cumulated_losses = defaultdict(lambda: 0.0)
                all_metrics.append(metrics)
                logger.log_metrics(metrics)

                if output_dir is not None:
                    metrics_path.write_text(json.dumps(all_metrics, indent=2))
                    torch.save(model, str(model_path))

                model.train()

            if step == max_steps:
                break

            batch = next(iterator)
            optimizer.zero_grad()
            model.reset_cache()

            loss = torch.zeros((), device=accelerator.device)
            for component in trained_pipes:
                output = component.module_forward(
                    batch[component.name],
                    supervision=True,
                )
                loss += output["loss"]
                for key, value in output.items():
                    if key.endswith("loss"):
                        cumulated_losses[key] += float(value)

            accelerator.backward(loss)
            optimizer.step()

        if return_model:
            return model


if __name__ == "__main__":
    app()
