from typing import TYPE_CHECKING

from edspdf.utils.lazy_module import lazify

lazify()

if TYPE_CHECKING:
    from .simple import execute_simple_backend
    from .multiprocessing import execute_multiprocessing_backend
