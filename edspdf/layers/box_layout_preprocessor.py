from collections import defaultdict
from typing import Any, Dict

import torch
from typing_extensions import TypedDict

from edspdf import Module, registry
from edspdf.models import PDFDoc
from edspdf.utils.collections import flatten
from edspdf.utils.torch import pad_2d


class BoxBatch(TypedDict):
    page: torch.LongTensor
    xmin: torch.FloatTensor
    ymin: torch.FloatTensor
    xmax: torch.FloatTensor
    ymax: torch.FloatTensor
    width: torch.FloatTensor
    height: torch.FloatTensor
    first_page: torch.BoolTensor
    last_page: torch.BoolTensor
    page_ids: torch.LongTensor


@registry.factory.register("box-preprocessor")
class BoxLayoutPreprocessor(Module[PDFDoc, BoxBatch]):
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

    def __new__(cls):
        if BoxLayoutPreprocessor.INSTANCE is None:
            BoxLayoutPreprocessor.INSTANCE = super().__new__(cls)
        return BoxLayoutPreprocessor.INSTANCE

    def preprocess_boxes(self, boxes):
        box_pages = [box.page for box in boxes]

        last_page = max(box_pages, default=0)

        return {
            "page": [b.page for b in boxes],
            "xmin": [b.x0 for b in boxes],
            "ymin": [b.y0 for b in boxes],
            "xmax": [b.x1 for b in boxes],
            "ymax": [b.y1 for b in boxes],
            "width": [(b.x1 - b.x0) for b in boxes],
            "height": [(b.y1 - b.y0) for b in boxes],
            "first_page": [b.page == 0 for b in boxes],
            "last_page": [b.page == last_page for b in boxes],
        }

    def preprocess(self, doc: PDFDoc, supervision: bool = False):
        return self.preprocess_boxes(doc.lines)

    def collate(self, batch, device: torch.device) -> BoxBatch:
        page_boxes_id = defaultdict(lambda: [])
        doc_pages = [[] for _ in range(len(batch["page"]))]
        box_i = 0
        for doc_i, doc_boxes_page in enumerate(batch["page"]):
            for box_page in doc_boxes_page:
                page_boxes_id[(doc_i, box_page)].append(box_i)
                doc_pages[doc_i].append(box_i)
                box_i += 1

        page_boxes_id = pad_2d(list(page_boxes_id.values()), pad=-1, device=device)

        (
            boxes_page,
            boxes_xmin,
            boxes_ymin,
            boxes_xmax,
            boxes_ymax,
            boxes_w,
            boxes_h,
            boxes_first_page,
            boxes_last_page,
        ) = torch.as_tensor(
            [
                flatten(batch["page"]),
                flatten(batch["xmin"]),
                flatten(batch["ymin"]),
                flatten(batch["xmax"]),
                flatten(batch["ymax"]),
                flatten(batch["width"]),
                flatten(batch["height"]),
                flatten(batch["first_page"]),
                flatten(batch["last_page"]),
            ],
            dtype=torch.float,
            device=device,
        ).unbind(
            0
        )

        return {
            "page": boxes_page.long(),
            "xmin": boxes_xmin,
            "ymin": boxes_ymin,
            "xmax": boxes_xmax,
            "ymax": boxes_ymax,
            "width": boxes_w,
            "height": boxes_h,
            "first_page": boxes_first_page,
            "last_page": boxes_last_page,
            "page_ids": page_boxes_id.long(),
        }

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        pass
