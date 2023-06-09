import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from weakref import WeakKeyDictionary

import catalogue
from confit import Config, Registry
from confit.registry import RegistryCollection

import edspdf
from edspdf.utils.collections import FrozenDict, FrozenList

PIPE_META = WeakKeyDictionary()


def accepted_arguments(
    func: Callable,
    args: Sequence[str],
) -> List[str]:
    """
    Checks that a function accepts a list of keyword arguments

    Parameters
    ----------
    func: Callable[..., T]
        Function to check
    args: Union[str, Sequence[str]]
        Argument or list of arguments to check

    Returns
    -------
    List[str]
    """
    sig = inspect.signature(func)
    has_kwargs = any(
        param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
    )
    if has_kwargs:
        return list(args)
    return [arg for arg in args if arg in sig.parameters]


@dataclass
class FactoryMeta:
    assigns: Iterable[str]
    requires: Iterable[str]
    retokenizes: bool
    default_config: Dict


class CurriedFactory:
    def __init__(self, func, kwargs):
        self.kwargs = kwargs
        self.factory = func
        # self.factory_name = factory_name
        self.instantiated = None

    def instantiate(
        obj: Any,
        pipeline: "edspdf.pipeline.Pipeline",
        path: Sequence[str] = (),
    ):
        """
        To ensure compatibility with spaCy's API, we need to support
        passing in the pipeline object and name to factories. Since they can be
        nested, we need to add them to every factory in the config.
        """
        if isinstance(obj, CurriedFactory):
            if obj.instantiated is not None:
                return obj.instantiated

            name = ".".join(path)

            kwargs = {
                key: CurriedFactory.instantiate(value, pipeline, (*path, key))
                for key, value in obj.kwargs.items()
            }
            obj.instantiated = obj.factory(
                **{
                    "pipeline": pipeline,
                    "name": name,
                    **kwargs,
                }
            )
            # Config._store_resolved(
            #     obj.instantiated,
            #     Config(
            #         {
            #             "@factory": obj.factory_name,
            #             **kwargs,
            #         }
            #     ),
            # )
            # PIPE_META[obj.instantiated] = obj.meta
            return obj.instantiated
        elif isinstance(obj, dict):
            return {
                key: CurriedFactory.instantiate(value, pipeline, (*path, key))
                for key, value in obj.items()
            }
        elif isinstance(obj, tuple):
            return tuple(
                CurriedFactory.instantiate(value, pipeline, (*path, str(i)))
                for i, value in enumerate(obj)
            )
        elif isinstance(obj, list):
            return [
                CurriedFactory.instantiate(value, pipeline, (*path, str(i)))
                for i, value in enumerate(obj)
            ]
        else:
            return obj


class FactoryRegistry(Registry):
    """
    A registry that validates the input arguments of the registered functions.
    """

    def get(self, name: str) -> Any:
        """Get the registered function for a given name.

        name (str): The name.
        RETURNS (Any): The registered function.
        """

        def curried(**kwargs):
            return CurriedFactory(func, kwargs=kwargs)

        namespace = list(self.namespace) + [name]
        if catalogue.check_exists(*namespace):
            func = catalogue._get(namespace)
            return curried

        if self.entry_points:
            self.get_entry_point(name)
            if catalogue.check_exists(*namespace):
                func = catalogue._get(namespace)
                return curried

        available = self.get_available()
        current_namespace = " -> ".join(self.namespace)
        available_str = ", ".join(available) or "none"
        raise catalogue.RegistryError(
            f"Can't find '{name}' in registry {current_namespace}. "
            f"Available names: {available_str}"
        )

    def register(
        self,
        name: str,
        *,
        func: Optional[catalogue.InFunc] = None,
        default_config: Dict[str, Any] = FrozenDict(),
        assigns: Iterable[str] = FrozenList(),
        requires: Iterable[str] = FrozenList(),
        retokenizes: bool = False,
        default_score_weights: Dict[str, Optional[float]] = FrozenDict(),
    ) -> Callable[[catalogue.InFunc], catalogue.InFunc]:
        """
        This is a convenience wrapper around `confit.Registry.register`, that
        curries the function to be registered, allowing to instantiate the class
        later once `pipeline` and `name` are known.

        Parameters
        ----------
        name: str
        func: Optional[catalogue.InFunc]
        default_config: Dict[str, Any]
        assigns: Iterable[str]
        requires: Iterable[str]
        retokenizes: bool
        default_score_weights: Dict[str, Optional[float]]

        Returns
        -------
        Callable[[catalogue.InFunc], catalogue.InFunc]
        """
        save_params = {"@factory": name}

        def register(fn: catalogue.InFunc) -> catalogue.InFunc:
            if len(accepted_arguments(fn, ["pipeline", "name"])) < 2:
                raise ValueError(
                    "Factory functions must accept pipeline and name as arguments."
                )

            meta = FactoryMeta(
                assigns=assigns,
                requires=requires,
                retokenizes=retokenizes,
                default_config=default_config,
            )

            def invoke(validated_fn, kwargs):
                if default_config is not None:
                    kwargs = (
                        Config(default_config)
                        .resolve(registry=self.registry)
                        .merge(kwargs)
                    )
                instantiated = validated_fn(kwargs)
                PIPE_META[instantiated] = meta
                return instantiated

            registered_fn = Registry.register(
                self,
                name=name,
                save_params=save_params,
                skip_save_params=["pipeline", "name"],
                func=fn,
                invoker=invoke,
            )

            return registered_fn

        return register(func) if func is not None else register


class registry(RegistryCollection):
    factory = factories = FactoryRegistry(("edspdf", "factories"), entry_points=True)
    misc = Registry(("edspdf", "misc"), entry_points=True)
    adapter = Registry(("edspdf", "adapter"), entry_points=True)
