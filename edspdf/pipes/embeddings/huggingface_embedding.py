import math
import sys
from typing import Optional, Set

import torch
from confit import validate_arguments
from foldedtensor import as_folded_tensor
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig as BitsAndBytesConfig_
from typing_extensions import Literal

from edspdf import TrainablePipe, registry
from edspdf.pipeline import Pipeline
from edspdf.pipes.embeddings import EmbeddingOutput
from edspdf.structures import PDFDoc

BitsAndBytesConfig = validate_arguments(BitsAndBytesConfig_)


def compute_contextualization_scores(windows):
    ramp = torch.arange(0, windows.shape[1], 1)
    scores = (
        torch.min(ramp, windows.mask.sum(1, keepdim=True) - 1 - ramp)
        .clamp(min=0)
        .view(-1)
    )
    return scores


@registry.factory.register("huggingface-embedding")
class HuggingfaceEmbedding(TrainablePipe[EmbeddingOutput]):
    """
    The HuggingfaceEmbeddings component is a wrapper around the Huggingface multi-modal
    models. Such pre-trained models should offer better results than a model trained
    from scratch. Compared to using the raw Huggingface model, we offer a simple
    mechanism to split long documents into strided windows before feeding them to the
    model.

    ## Windowing

    The HuggingfaceEmbedding component splits long documents into smaller windows before
    feeding them to the model. This is done to avoid hitting the maximum number of
    tokens that can be processed by the model on a single device. The window size and
    stride can be configured using the `window` and `stride` parameters. The default
    values are 510 and 255 respectively, which means that the model will process windows
    of 510 tokens, each separated by 255 tokens. Whenever a token appears in multiple
    windows, the embedding of the "most contextualized" occurrence is used, i.e. the
    occurrence that is the closest to the center of its window.

    Here is an overview how this works in a classifier model :
    ![Transformer windowing](/assets/images/transformer-windowing.svg)

    Examples
    --------

    Here is an example of how to define a pipeline with the HuggingfaceEmbedding
    component:

    ```python
    from edspdf import Pipeline

    model = Pipeline()
    model.add_pipe(
        "pdfminer-extractor",
        name="extractor",
        config={
            "render_pages": True,
        },
    )
    model.add_pipe(
        "huggingface-embedding",
        name="embedding",
        config={
            "model": "microsoft/layoutlmv3-base",
            "use_image": False,
            "window": 128,
            "stride": 64,
            "line_pooling": "mean",
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
    ```

    This model can then be trained following the
    [training recipe](/recipes/training/).

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline instance
    name: str
        The component name
    model: str
        The Huggingface model name or path
    use_image: bool
        Whether to use the image or not in the model
    window: int
        The window size to use when splitting long documents into smaller windows
        before feeding them to the Transformer model (default: 510 = 512 - 2)
    stride: int
        The stride (distance between windows) to use when splitting long documents into
        smaller windows: (default: 510 / 2 = 255)
    line_pooling: Literal["mean", "max", "sum"]
        The pooling strategy to use when combining the embeddings of the tokens in a
        line into a single line embedding
    max_tokens_per_device: int
        The maximum number of tokens that can be processed by the model on a single
        device. This does not affect the results but can be used to reduce the memory
        usage of the model, at the cost of a longer processing time.
    quantization_config: Optional[BitsAndBytesConfig]
        The quantization configuration to use when loading the model
    kwargs:
        Additional keyword arguments to pass to the Huggingface
        `AutoModel.from_pretrained` method
    """

    def __init__(
        self,
        pipeline: Pipeline = None,
        name: str = "huggingface-embedding",
        model: str = None,
        use_image: bool = True,
        window: int = 510,
        stride: int = 255,
        line_pooling: Literal["mean", "max", "sum"] = "mean",
        max_tokens_per_device: int = sys.maxsize,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        **kwargs,
    ):
        super().__init__(pipeline, name)
        self.use_image = use_image
        self.image_processor = (
            AutoImageProcessor.from_pretrained(model, apply_ocr=False)
            if use_image
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.hf_model = AutoModel.from_pretrained(
            model,
            quantization_config=quantization_config,
            **kwargs,
        )
        self.output_size = self.hf_model.config.hidden_size
        self.window = window
        self.stride = stride
        self.line_pooling = line_pooling
        self.max_tokens_per_device = max_tokens_per_device

    def preprocess(self, doc: PDFDoc):
        res = {
            "input_ids": [],
            "bbox": [],
            "windows": [],
            "line_starts": [],
        }
        if self.use_image:
            res["pixel_values"] = []

        for page in doc.pages:
            # Preprocess it using LayoutLMv3
            width = page.width
            height = page.height

            ratio = width / height
            width, height = (1000, 1000 / ratio) if width > 1000 else (width, height)
            width, height = (1000 * ratio, 1000) if height > 1000 else (width, height)

            prep = self.tokenizer(
                text=[line.text for line in page.text_boxes],
                boxes=[
                    (
                        int(line.x0 * width),
                        int(line.y0 * height),
                        int(line.x1 * width),
                        int(line.y1 * height),
                    )
                    for line in page.text_boxes
                ],
                word_labels=range(len(page.text_boxes)),
                return_attention_mask=True,
            )
            if self.use_image:
                prep.update(self.image_processor(images=page.image))

            # Compute line offsets into layoutlm generated tokens
            line_indices = prep["labels"][:-1]
            line_starts = [
                i
                for i, curr_index in enumerate(line_indices)
                if curr_index != -100 and curr_index != line_indices[i - 1]
            ]

            res["input_ids"].append(prep["input_ids"])
            res["bbox"].append(prep["bbox"])
            res["line_starts"].append(line_starts)
            if self.use_image:
                res["pixel_values"].append(prep["pixel_values"][0])

        return res

    def collate(self, batch):
        # Flatten most of these arrays to process batches page per page and
        # not sample per sample

        offset = 0
        window_max_size = 0
        window_count = 0
        windows = []
        windows_count_per_page = []
        for sample_input_ids in batch["input_ids"]:
            for page_input_ids in sample_input_ids:
                # fmt: off
                windows.append([
                    [
                        offset + 0,
                        *range(1 + offset + window_i * self.stride,
                               1 + offset + min(window_i * self.stride + self.window, len(page_input_ids) - 2)),  # noqa: E501
                        offset + len(page_input_ids) - 1,
                    ]
                    for window_i in range(0, 1 + max(0, math.ceil((len(page_input_ids) - 2 - self.window) / self.stride)))  # noqa: E501
                ])
                windows_count_per_page.append(len(windows[-1]))
                # fmt: on
                offset += len(page_input_ids)
                window_max_size = max(window_max_size, max(map(len, windows[-1])))
                window_count += len(windows[-1])

        windows = as_folded_tensor(
            windows,
            full_names=("page", "window", "token"),
            data_dims=("window", "token"),
            dtype=torch.long,
        )
        indexer = torch.zeros(
            (windows.max() + 1) if windows.numel() else 0, dtype=torch.long
        )

        # Sort each occurrence of an initial token by its contextualization score:
        # We can only use the amax reduction, so to retrieve the best occurrence, we
        # insert the index of the token output by the transformer inside the score
        # using a lexicographic approach
        # (score + index / n_tokens) ~ (score * n_tokens + index), taking the max,
        # and then retrieving the index of the token using the modulo operator.
        scores = compute_contextualization_scores(windows)
        scores = scores * len(scores) + torch.arange(len(scores))
        indexer.index_reduce_(
            dim=0,
            source=scores,
            index=windows.view(-1),
            reduce="amax",
        )
        indexer %= len(scores)

        # Get token indices for each line -> sample, page, line, token
        line_window_indices = []
        line_window_offsets_flat = [0]
        offset = 0
        for sample_input_ids, sample_line_starts in zip(
            batch["input_ids"], batch["line_starts"]
        ):
            sample_line_window_indices = []
            line_window_indices.append(sample_line_window_indices)
            for page_line_starts, page_input_ids in zip(
                sample_line_starts, sample_input_ids
            ):
                page_line_window_indices = []
                sample_line_window_indices.append(page_line_window_indices)
                for line_start, line_end in zip(
                    page_line_starts, (*page_line_starts[1:], len(page_input_ids))
                ):
                    line_window_offsets_flat.append(
                        line_window_offsets_flat[-1] + line_end - line_start
                    )
                    page_line_window_indices.append(
                        list(range(offset + line_start, offset + line_end))
                    )
                offset += len(page_input_ids)
        line_window_indices = as_folded_tensor(
            line_window_indices,
            full_names=("sample", "page", "line", "token"),
            data_dims=("token",),
            dtype=torch.long,
        )
        last_after_one = max(0, len(line_window_offsets_flat) - 1)
        line_window_offsets_flat = as_folded_tensor(
            # discard the last offset, since we start from 0 and add each line length
            data=torch.as_tensor(
                line_window_offsets_flat[:last_after_one], dtype=torch.long
            ),
            data_dims=("line",),
            full_names=("sample", "page", "line"),
            lengths=line_window_indices.lengths[:-1],
        )
        kw = dict(
            full_names=("sample", "page", "subword"),
            data_dims=("subword",),
        )
        collated = {
            "input_ids": as_folded_tensor(batch["input_ids"], **kw, dtype=torch.long),
            "bbox": as_folded_tensor(batch["bbox"], **kw, dtype=torch.long),
            "windows": windows,
            "indexer": indexer[line_window_indices],
            "line_window_indices": indexer[line_window_indices].as_tensor(),
            "line_window_offsets_flat": line_window_offsets_flat,
        }
        if self.use_image:
            collated["pixel_values"] = torch.as_tensor(
                [
                    page_pixels
                    for sample_pages in batch["pixel_values"]
                    for page_pixels in sample_pages
                ],
            ).repeat_interleave(
                torch.as_tensor(windows_count_per_page, dtype=torch.long), dim=0
            )
        return collated

    def forward(self, batch):
        if 0 in batch["input_ids"].shape:
            return {
                "embeddings": batch["line_window_offsets_flat"].view(
                    *batch["line_window_offsets_flat"].shape, self.output_size
                ),
            }

        windows = batch["windows"]
        kwargs = dict(
            input_ids=batch["input_ids"].as_tensor()[windows],
            bbox=batch["bbox"].as_tensor()[windows],
            attention_mask=windows.mask,
            pixel_values=(
                batch.get("pixel_values").to(next(self.parameters()).dtype)
                if self.use_image
                else None
            ),
        )
        num_windows_per_batch = self.max_tokens_per_device // (
            windows.shape[1]
            + (
                (self.hf_model.config.input_size // self.hf_model.config.patch_size)
                ** 2
                if self.use_image and hasattr(self.hf_model.config, "patch_size")
                else 0
            )
        )

        token_embeddings = [
            self.hf_model.forward(
                **{
                    k: None if v is None else v[offset : offset + num_windows_per_batch]
                    for k, v in kwargs.items()
                }
            ).last_hidden_state[:, : windows.shape[1]]
            # TODO offset line_window_indices during collate
            #      instead of slicing token_embeddings
            for offset in range(0, len(windows), num_windows_per_batch)
        ]
        token_embeddings = (
            torch.cat(token_embeddings, dim=0)
            if len(token_embeddings) > 1
            else token_embeddings[0]
        )
        line_embedding = torch.nn.functional.embedding_bag(
            input=batch["line_window_indices"],
            weight=token_embeddings.reshape(-1, token_embeddings.size(-1)),
            offsets=batch["line_window_offsets_flat"],
            mode=self.line_pooling,
        )
        return {"embeddings": line_embedding}

    def to_disk(self, path, *, exclude: Optional[Set[str]]):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        for obj in (self.tokenizer, self.image_processor, self.hf_model):
            if obj is not None:
                obj.save_pretrained(path)
        for param in self.hf_model.parameters():
            exclude.add(object.__repr__(param))
        cfg = super().to_disk(path, exclude=exclude) or {}
        cfg["model"] = f"./{path.as_posix()}"
        return cfg
