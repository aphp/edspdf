from typing import Any, Dict, Sequence

import torch
from foldedtensor import FoldedTensor, as_folded_tensor
from typing_extensions import TypedDict

from edspdf import Pipeline, TrainablePipe, registry
from edspdf.structures import PDFDoc, TextBox

BoxLayoutBatch = TypedDict(
    "BoxLayoutBatch",
    {
        "page": FoldedTensor,
        "xmin": FoldedTensor,
        "ymin": FoldedTensor,
        "xmax": FoldedTensor,
        "ymax": FoldedTensor,
        "width": FoldedTensor,
        "height": FoldedTensor,
        "first_page": FoldedTensor,
        "last_page": FoldedTensor,
    },
)


@registry.factory.register("box-layout-preprocessor")
class BoxLayoutPreprocessor(TrainablePipe[BoxLayoutBatch]):
    """
    The box preprocessor is singleton since its is not configurable.
    The following features of each box of an input PDFDoc document are encoded
    as 1D tensors:

    - `boxes_page`: page index of the box
    - `boxes_first_page`: is the box on the first page
    - `boxes_last_page`: is the box on the last page
    - `boxes_xmin`: left position of the box
    - `boxes_ymin`: bottom position of the box
    - `boxes_xmax`: right position of the box
    - `boxes_ymax`: top position of the box
    - `boxes_w`: width position of the box
    - `boxes_h`: height position of the box

    The preprocessor also returns an additional tensors:

    - `page_boxes_id`: box indices per page to index the
      above 1D tensors (LongTensor: n_pages * n_boxes)
    """

    INSTANCE = None

    def __new__(cls, *args, **kwargs):
        if BoxLayoutPreprocessor.INSTANCE is None:
            BoxLayoutPreprocessor.INSTANCE = super().__new__(cls)
        return BoxLayoutPreprocessor.INSTANCE

    def __init__(
        self,
        pipeline: Pipeline = None,
        name: str = "box-layout-preprocessor",
    ):
        super().__init__(pipeline, name)

    def preprocess_boxes(self, boxes: Sequence[TextBox]):
        box_pages = [box.page.page_num for box in boxes]

        last_page = max(box_pages, default=0)

        return {
            "page": box_pages,
            "xmin": [b.x0 for b in boxes],
            "ymin": [b.y0 for b in boxes],
            "xmax": [b.x1 for b in boxes],
            "ymax": [b.y1 for b in boxes],
            "width": [(b.x1 - b.x0) for b in boxes],
            "height": [(b.y1 - b.y0) for b in boxes],
            "first_page": [b.page_num == 0 for b in boxes],
            "last_page": [b.page_num == last_page for b in boxes],
        }

    def preprocess(self, doc: PDFDoc, supervision: bool = False):
        pages = doc.pages
        box_pages = [[b.page.page_num for b in page.text_boxes] for page in pages]
        last_page = max(box_pages, default=0)
        return {
            "page": box_pages,
            "xmin": [[b.x0 for b in p.text_boxes] for p in pages],
            "ymin": [[b.y0 for b in p.text_boxes] for p in pages],
            "xmax": [[b.x1 for b in p.text_boxes] for p in pages],
            "ymax": [[b.y1 for b in p.text_boxes] for p in pages],
            "width": [[(b.x1 - b.x0) for b in p.text_boxes] for p in pages],
            "height": [[(b.y1 - b.y0) for b in p.text_boxes] for p in pages],
            "first_page": [[b.page.page_num == 0 for b in p.text_boxes] for p in pages],
            "last_page": [
                [b.page.page_num == last_page for b in p.text_boxes] for p in pages
            ],
        }

    def collate(self, batch, device: torch.device) -> BoxLayoutBatch:
        kw = {
            "full_names": ["sample", "page", "line"],
            "data_dims": ["line"],
            "device": device,
        }

        return {
            "page": as_folded_tensor(batch["page"], dtype=torch.long, **kw),
            "xmin": as_folded_tensor(batch["xmin"], dtype=torch.long, **kw),
            "ymin": as_folded_tensor(batch["ymin"], dtype=torch.long, **kw),
            "xmax": as_folded_tensor(batch["xmax"], dtype=torch.long, **kw),
            "ymax": as_folded_tensor(batch["ymax"], dtype=torch.long, **kw),
            "width": as_folded_tensor(batch["width"], dtype=torch.long, **kw),
            "height": as_folded_tensor(batch["height"], dtype=torch.long, **kw),
            "first_page": as_folded_tensor(batch["first_page"], dtype=torch.bool, **kw),
            "last_page": as_folded_tensor(batch["last_page"], dtype=torch.bool, **kw),
        }

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        pass
