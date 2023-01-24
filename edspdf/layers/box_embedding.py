from typing import Optional

import torch

from edspdf import Module, registry


@registry.factory.register("box-embedding")
class BoxEmbedding(Module):
    def __init__(
        self,
        size: int,
        dropout_p: float = 0.2,
        layout_encoder: Optional[Module] = {},
        text_encoder: Optional[Module] = None,
        contextualizer: Optional[Module] = None,
    ):
        """
        Encodes boxes using a combination of layout and text features.

        Parameters
        ----------
        size: int
            Size of the output box embedding
        dropout_p: float
            Dropout probability used on the output of the box and textual encoders
        text_encoder: Dict
            Config for the text encoder
        layout_encoder: Dict
            Config for the layout encoder
        """
        super().__init__()

        assert size % 6 == 0, "Embedding dimension must be dividable by 6"

        self.size = size

        self.layout_encoder = layout_encoder
        self.text_encoder = text_encoder
        self.contextualizer = contextualizer
        self.dropout = torch.nn.Dropout(dropout_p)

    @property
    def output_size(self):
        return self.size

    def initialize(self, gold_data, **kwargs):
        super().initialize(gold_data, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.initialize(gold_data, size=self.size)
        if self.layout_encoder is not None:
            self.layout_encoder.initialize(gold_data, size=self.size)
        if self.contextualizer is not None:
            self.contextualizer.initialize(gold_data, input_size=self.size)

    def preprocess(self, doc, supervision: bool = False):
        return {
            "boxes": self.layout_encoder.preprocess(doc, supervision=supervision)
            if self.layout_encoder is not None
            else None,
            "texts": self.text_encoder.preprocess(doc, supervision=supervision)
            if self.text_encoder is not None
            else None,
        }

    def collate(self, batch, device: torch.device):
        return {
            "texts": self.text_encoder.collate(batch["texts"], device)
            if self.text_encoder is not None
            else None,
            "boxes": self.layout_encoder.collate(batch["boxes"], device)
            if self.layout_encoder is not None
            else None,
        }

    def forward(self, batch, supervision=False):
        embeds = sum(
            [
                self.dropout(encoder.module_forward(batch[name]))
                for name, encoder in (
                    ("boxes", self.layout_encoder),
                    ("texts", self.text_encoder),
                )
                if encoder is not None
            ]
        )

        if self.contextualizer is not None:
            embeds = self.contextualizer(
                embeds=embeds,
                boxes=batch["boxes"],
            )

        return embeds
