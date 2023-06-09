import torch
from typing_extensions import Literal

from edspdf import Pipeline, registry
from edspdf.pipes.embeddings import EmbeddingOutput, TrainablePipe


@registry.factory.register("embedding-combiner")
class EmbeddingCombiner(TrainablePipe[EmbeddingOutput]):
    def __init__(
        self,
        dropout_p: float = 0.0,
        mode: Literal["sum", "cat"] = "sum",
        pipeline: Pipeline = None,
        name: str = "embedding-combiner",
        **encoders: TrainablePipe[EmbeddingOutput],
    ):
        """
        Encodes boxes using a combination of multiple encoders

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline object
        name: str
            The name of the pipe
        mode: Literal["sum", "cat"]
            The mode to use to combine the encoders:

            - `sum`: Sum the outputs of the encoders
            - `cat`: Concatenate the outputs of the encoders
        dropout_p: float
            Dropout probability used on the output of the box and textual encoders
        encoders: Dict[str, TrainablePipe[EmbeddingOutput]]
            The encoders to use. The keys are the names of the encoders and the values
            are the encoders themselves.
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
        all_embeds = [
            self.dropout(res["embeddings"].refold(results[0]["embeddings"].data_dims))
            for res in results
        ]
        embeddings = (
            sum(all_embeds) if self.mode == "sum" else torch.cat(all_embeds, dim=-1)
        )
        return {"embeddings": embeddings}  # type: ignore
