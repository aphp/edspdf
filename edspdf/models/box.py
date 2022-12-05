from typing import Optional

import attrs

from edspdf.model import BaseModel


class Box(BaseModel):
    page: int = None

    x0: float
    x1: float
    y0: float
    y1: float

    page_width: Optional[float] = None
    page_height: Optional[float] = None

    label: Optional[str] = None

    source: Optional[str] = None

    dict = attrs.asdict

    def __lt__(self, other):
        if self.page < other.page:
            return True
        if self.page > other.page:
            return False

        alpha = 0.2
        beta = 1 - alpha
        self_x0 = self.x0 * beta + self.x1 * alpha
        self_x1 = self.x0 * alpha + self.x1 * beta
        self_y0 = self.y0 * beta + self.y1 * alpha
        self_y1 = self.y0 * alpha + self.y1 * beta

        other_x0 = other.x0 * beta + other.x1 * alpha
        other_x1 = other.x0 * alpha + other.x1 * beta
        other_y0 = other.y0 * beta + other.y1 * alpha
        other_y1 = other.y0 * alpha + other.y1 * beta

        dy0 = other_y1 - self_y0
        dy1 = other_y0 - self_y1
        if dy0 > 0 and dy1 > 0:
            return True
        if dy0 < 0 and dy1 < 0:
            return False
        dx0 = other_x1 - self_x0
        dx1 = other_x0 - self_x1

        if dx0 > 0 and dx1 > 0:
            return True
        if dx0 < 0 and dx1 < 0:
            return False

        return ((self.y0 + self.y1) / 2, (self.x0 + self.x1) / 2) < (
            (other.y0 + other.y1) / 2,
            (other.x0 + other.x1) / 2,
        )
