from typing import List, Optional

import attrs

from edspdf.model import BaseModel

from .text_box import TextBox


class PDFDoc(BaseModel):
    id: str = None
    content: bytes = attrs.field(repr=lambda c: f"{len(c)} bytes")
    text: Optional[str] = None
    lines: List[TextBox] = []
    error: bool = False
