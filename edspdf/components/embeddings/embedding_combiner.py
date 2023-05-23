import torch

from edspdf import Pipeline, TorchComponent, registry
from edspdf.components.embeddings import BoxEmbeddingComponent, EmbeddingOutput


@registry.factory.register("embedding_combiner")
class EmbeddingCombinerComponent(TorchComponent[EmbeddingOutput]):
    def __init__(
        self,
        dropout_p: float = 0.2,
        mode: str = "sum",
        pipeline: Pipeline = None,
        name: str = "embedding_combiner",
        **encoders: BoxEmbeddingComponent,
    ):
        """
        Encodes boxes using a combination of layout and text features.

        Parameters
        ----------
        dropout_p: float
            Dropout probability used on the output of the box and textual encoders
        text_encoder: Dict
            Config for the text encoder
        layout_encoder: Dict
            Config for the layout encoder
        """
        super().__init__(pipeline, name)

        for name, encoder in encoders.items():
            setattr(self, name, encoder)

        self.mode = mode

        assert (
            mode != "sum"
            or len(set(encoder.output_size for encoder in encoders.values())) == 1
        ), (
            "All encoders must have the same output size when using 'sum' "
            "combination:\n{}".format(
                "\n".join(
                    "- {}: {}".format(name, encoder.output_size)
                    for name, encoder in encoders.items()
                )
            )
        )

        self.dropout = torch.nn.Dropout(dropout_p)
        self.output_size = (
            sum(encoder.output_size for encoder in encoders.values())
            if mode == "cat"
            else next(iter(encoders.values())).output_size
        )

    def forward(self, batch) -> EmbeddingOutput:
        results = [
            encoder.module_forward(batch[name])
            for name, encoder in self.named_component_children()
        ]
        all_embeds = [self.dropout(res["embeddings"]) for res in results]
        embeddings = (
            sum(all_embeds) if self.mode == "sum" else torch.cat(all_embeds, dim=-1)
        )
        return {"embeddings": embeddings}  # type: ignore
