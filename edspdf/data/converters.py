"""
Converters are used to convert documents between python dictionaries and Doc objects.
There are two types of converters: readers and writers. Readers convert dictionaries to
Doc objects, and writers convert Doc objects to dictionaries.
"""
import inspect
from copy import copy
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from confit.registry import ValidatedFunction

FILENAME = "__FILENAME__"
CONTENT = "__CONTENT__"

SCHEMA = {}


def validate_kwargs(converter, kwargs):
    converter: FunctionType = copy(converter)
    spec = inspect.getfullargspec(converter)
    first = spec.args[0]
    converter.__annotations__[first] = Optional[Any]
    converter.__defaults__ = (None, *(spec.defaults or ())[-len(spec.args) + 1 :])
    vd = ValidatedFunction(converter, {"arbitrary_types_allowed": True})
    model = vd.init_model_instance(**kwargs)
    d = {
        k: v
        for k, v in model._iter()
        if (k in model.__fields__ or model.__fields__[k].default_factory)
    }
    d.pop("v__duplicate_kwargs", None)  # see pydantic ValidatedFunction code
    d.pop(vd.v_args_name, None)
    d.pop(first, None)
    return {**(d.pop(vd.v_kwargs_name, None) or {}), **d}


def get_dict2doc_converter(converter: Callable, kwargs) -> Tuple[Callable, Dict]:
    # kwargs_to_init = False
    # if not callable(converter):
    #     available = edspdf.registry.factory.get_available()
    #     try:
    #         filtered = [
    #             name
    #             for name in available
    #             if converter == name or (converter in name and "dict2doc" in name)
    #         ]
    #         converter = edspdf.registry.factory.get(filtered[0])
    #         converter = converter(**kwargs).instantiate(nlp=None)
    #         kwargs = {}
    #         return converter, kwargs
    #     except (KeyError, IndexError):
    #         available = [v for v in available if "dict2doc" in v]
    #         raise ValueError(
    #             f"Cannot find converter for format {converter}. "
    #             f"Available converters are {', '.join(available)}"
    #         )
    # if isinstance(converter, type) or kwargs_to_init:
    #     return converter(**kwargs), {}
    return converter, validate_kwargs(converter, kwargs)


def get_doc2dict_converter(converter: Callable, kwargs) -> Tuple[Callable, Dict]:
    # if not callable(converter):
    #     available = edspdf.registry.factory.get_available()
    #     try:
    #         filtered = [
    #             name
    #             for name in available
    #             if converter == name or (converter in name and "doc2dict" in name)
    #         ]
    #         converter = edspdf.registry.factory.get(filtered[0])
    #         converter = converter(**kwargs).instantiate(nlp=None)
    #         kwargs = {}
    #         return converter, kwargs
    #     except (KeyError, IndexError):
    #         available = [v for v in available if "doc2dict" in v]
    #         raise ValueError(
    #             f"Cannot find converter for format {converter}. "
    #             f"Available converters are {', '.join(available)}"
    #         )
    return converter, validate_kwargs(converter, kwargs)
