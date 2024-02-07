from typing import TYPE_CHECKING
from edspdf.utils.lazy_module import lazify

lazify()

if TYPE_CHECKING:
    from .base import from_iterable, to_iterable
    from .files import read_files, write_files
    from .parquet import read_parquet, write_parquet
    from .pandas import from_pandas, to_pandas
    from .converters import get_dict2doc_converter, get_doc2dict_converter
