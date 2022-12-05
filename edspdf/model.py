from typing import TYPE_CHECKING

import attrs

from .config import Config


class BaseModelMeta(type):
    def __new__(mcs, name, bases, class_dict):
        cls = super().__new__(mcs, name, bases, class_dict)
        if "__attrs_attrs__" in class_dict:
            # Avoid infinite recursion since attrs performs the following call
            # inside define
            # > type(cls)(name, bases, new_class_dict)
            # triggering this method again.
            return cls
        return attrs.define(kw_only=True, hash=True)(cls)


class BaseModel(metaclass=BaseModelMeta):
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, dict):
            v = Config(v).resolve(deep=False)
        if isinstance(v, dict):
            return cls(**v)
        if isinstance(v, cls):
            return v
        raise ValueError(f"Could not cast {cls.__name__} from {v}")

    def dict(self):
        return attrs.asdict(self)

    def copy(self):
        return attrs.evolve(self)

    def evolve(self, **update):
        return attrs.evolve(self, **update)


if TYPE_CHECKING:
    import pydantic

    class BaseModel(pydantic.BaseModel):  # noqa: F811
        pass
