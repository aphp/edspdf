from typing import Any, Dict, Optional

import torch

from edspdf import Module, registry
from edspdf.utils.torch import compute_pdf_relative_positions

from .block_transformer import BlockTransformer
from .box_embedding import BoxEmbedding
from .text_embedding import TextEmbedding


@registry.factory.register("text-box-embedding")
class TextBoxEmbedding(Module):
    def __init__(
        self,
        size: int,
        n_relative_positions: Optional[int] = None,
        dropout_p: float = 0.2,
        box_encoder: Dict[str, Any] = {},
        text_encoder: Dict[str, Any] = {},
        contextualizer: Dict[str, Any] = None,
    ):
        """

        Parameters
        ----------
        size: int
            Size of the output box embedding
        n_relative_positions: int
            Maximum range of embeddable relative positions between boxes (further
            distances are capped to ±n_relative_positions // 2)
        dropout_p: float
            Dropout probability used on the output of the box and textual encoders
        text_encoder: torch.nn.Module
            The module used to encode the textual features of the boxes
        box_encoder: Dict
            Parameters of the [BoxEmbedding] layer
        """
        super().__init__()

        assert size % 6 == 0, "Embedding dimension must be dividable by 6"

        self.size = size

        self.box_encoder = BoxEmbedding(
            size=size,
            **box_encoder,
        )
        self.text_encoder = TextEmbedding(
            size=size,
            **text_encoder,
        )
        self.empty_embed = torch.nn.Parameter(torch.randn(size))
        if contextualizer is not None:
            self.n_relative_positions = n_relative_positions
            self.relative_position_embedding = torch.nn.Parameter(
                torch.randn(
                    (
                        n_relative_positions,
                        size,
                    )
                )
            )
            self.contextualizer = BlockTransformer(
                input_size=size,
                position_embedding=self.relative_position_embedding,
                **contextualizer,
            )
        else:
            self.contextualizer = None
            self.relative_position_embedding = None
            self.n_relative_positions = None
        self.dropout = torch.nn.Dropout(dropout_p)

    @property
    def output_size(self):
        return self.size

    def initialize(self, gold_data):
        self.text_encoder.initialize(gold_data)
        self.box_encoder.initialize(gold_data)

    def preprocess(self, doc, supervision: bool = False):
        return {
            "boxes": self.box_encoder.preprocess(doc, supervision=supervision),
            "texts": self.text_encoder.preprocess(doc, supervision=supervision),
        }

    def collate(self, batch, device: torch.device):
        self.last_prep = batch

        return {
            "texts": self.text_encoder.collate(batch["texts"], device),
            "boxes": self.box_encoder.collate(batch["boxes"], device),
        }

    def forward(self, batch, supervision=False):
        self.last_batch = batch

        box_embeds = self.box_encoder.module_forward(batch["boxes"])
        text_embeds = self.text_encoder.module_forward(batch["texts"])
        embeds = self.dropout(box_embeds) + self.dropout(text_embeds)

        if self.contextualizer is not None:
            page_boxes = batch["boxes"]["page_ids"]
            page_boxes_mask = page_boxes != -1

            n_pages = page_boxes.shape[0]
            device = page_boxes.device
            embeds_with_cls = torch.cat(
                [
                    self.empty_embed * torch.ones(n_pages, 1, 1, device=device),
                    embeds[page_boxes],
                ],
                dim=1,
            )
            mask_with_cls = torch.cat(
                [
                    torch.ones(page_boxes.shape[0], 1, dtype=torch.bool, device=device),
                    page_boxes_mask,
                ],
                dim=1,
            )
            n = page_boxes.shape[1] + 1
            positions_with_cls = torch.zeros(
                n_pages, n, n, 2, dtype=torch.long, device=device
            )
            positions_with_cls[:, 1:, 1:, :] = compute_pdf_relative_positions(
                x0=batch["boxes"]["xmin"][page_boxes],
                x1=batch["boxes"]["xmax"][page_boxes],
                y0=batch["boxes"]["ymin"][page_boxes],
                y1=batch["boxes"]["ymax"][page_boxes],
                width=batch["boxes"]["width"][page_boxes],
                height=batch["boxes"]["height"][page_boxes],
                n_relative_positions=self.n_relative_positions,
            )
            no_position = torch.ones(n_pages, n, n, dtype=torch.bool, device=device)
            no_position[:, 1:, 1:] = 0

            embeds = self.contextualizer(
                embeds=embeds_with_cls,
                mask=mask_with_cls,
                relative_positions=positions_with_cls,
                no_position_mask=no_position,
            )[0][:, 1:][page_boxes_mask]

        return embeds
