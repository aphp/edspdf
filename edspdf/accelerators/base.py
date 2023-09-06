from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Union

from ..structures import PDFDoc


class FromDictFieldsToDoc:
    def __init__(self, content_field, id_field=None):
        self.content_field = content_field
        self.id_field = id_field

    def __call__(self, item):
        if isinstance(item, dict):
            return PDFDoc(
                content=item[self.content_field],
                id=item[self.id_field] if self.id_field else None,
            )
        return item


class ToDoc:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, config=None):
        if isinstance(value, str):
            return FromDictFieldsToDoc(value)
        elif isinstance(value, dict):
            return FromDictFieldsToDoc(**value)
        elif callable(value):
            return value
        else:
            raise TypeError(
                f"Invalid entry {value} ({type(value)}) for ToDoc, "
                f"expected string, a dict or a callable."
            )


def identity(x):
    return x


FROM_DOC_TO_DICT_FIELDS_TEMPLATE = """\
def fn(doc):
    return {X}
"""


class FromDocToDictFields:
    def __init__(self, mapping):
        self.mapping = mapping
        dict_fields = ", ".join(f"{k}: doc.{v}" for k, v in mapping.items())
        self.fn = eval(FROM_DOC_TO_DICT_FIELDS_TEMPLATE.replace("X", dict_fields))

    def __reduce__(self):
        return FromDocToDictFields, (self.mapping,)

    def __call__(self, doc):
        return self.fn(doc)


class FromDoc:
    """
    A FromDoc converter (from a PDFDoc to an arbitrary type) can be either:

    - a dict mapping field names to doc attributes
    - a callable that takes a PDFDoc and returns an arbitrary type
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, config=None):
        if isinstance(value, dict):
            return FromDocToDictFields(value)
        elif callable(value):
            return value
        else:
            raise TypeError(
                f"Invalid entry {value} ({type(value)}) for ToDoc, "
                f"expected dict or callable"
            )


class Accelerator:
    def __call__(
        self,
        inputs: Iterable[Any],
        model: Any,
        to_doc: ToDoc = FromDictFieldsToDoc("content"),
        from_doc: FromDoc = lambda doc: doc,
        component_cfg: Dict[str, Dict[str, Any]] = None,
    ):
        raise NotImplementedError()


if TYPE_CHECKING:
    ToDoc = Union[str, Dict[str, Any], Callable[[Any], PDFDoc]]  # noqa: F811
    FromDoc = Union[Dict[str, Any], Callable[[PDFDoc], Any]]  # noqa: F811
