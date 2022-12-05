from typing import Optional, Tuple

from .box import Box
from .style import SpannedStyle


class TextBox(Box):
    styles: Tuple[SpannedStyle] = []
    text: Optional[str] = None
    next_box: Optional["TextBox"] = None
