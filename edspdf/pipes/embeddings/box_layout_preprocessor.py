from typing import Any, Dict

import torch
from foldedtensor import FoldedTensor, as_folded_tensor
from typing_extensions import TypedDict

from edspdf import Pipeline, TrainablePipe, registry
from edspdf.structures import PDFDoc

BoxLayoutBatch = TypedDict(
    "BoxLayoutBatch",
    {
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

    def preprocess(self, doc: PDFDoc, supervision: bool = False):
        pages = doc.pages
        [[b.page_num for b in page.text_boxes] for page in pages]
        last_p = doc.num_pages - 1
        return {
            "xmin": [[b.x0 for b in p.text_boxes] for p in pages],
            "ymin": [[b.y0 for b in p.text_boxes] for p in pages],
            "xmax": [[b.x1 for b in p.text_boxes] for p in pages],
            "ymax": [[b.y1 for b in p.text_boxes] for p in pages],
            "width": [[(b.x1 - b.x0) for b in p.text_boxes] for p in pages],
            "height": [[(b.y1 - b.y0) for b in p.text_boxes] for p in pages],
            "first_page": [[b.page_num == 0 for b in p.text_boxes] for p in pages],
            "last_page": [[b.page_num == last_p for b in p.text_boxes] for p in pages],
        }

    def collate(self, batch) -> BoxLayoutBatch:
        kw = {
            "full_names": ["sample", "page", "line"],
            "data_dims": ["line"],
        }

        return {
            "xmin": as_folded_tensor(batch["xmin"], dtype=torch.float, **kw),
            "ymin": as_folded_tensor(batch["ymin"], dtype=torch.float, **kw),
            "xmax": as_folded_tensor(batch["xmax"], dtype=torch.float, **kw),
            "ymax": as_folded_tensor(batch["ymax"], dtype=torch.float, **kw),
            "width": as_folded_tensor(batch["width"], dtype=torch.float, **kw),
            "height": as_folded_tensor(batch["height"], dtype=torch.float, **kw),
            "first_page": as_folded_tensor(batch["first_page"], dtype=torch.bool, **kw),
            "last_page": as_folded_tensor(batch["last_page"], dtype=torch.bool, **kw),
        }

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()
