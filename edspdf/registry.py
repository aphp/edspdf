import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import catalogue
from confit import Config, Registry
from confit.errors import ConfitValidationError, patch_errors
from confit.registry import RegistryCollection

import edspdf
from edspdf.utils.collections import FrozenDict, FrozenList


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


class CurriedFactory:
    def __init__(self, func, kwargs):
        self.kwargs = kwargs
        self.factory = func
        self.instantiated = None
        self.error = None

    def instantiate(
        obj: Any,
        pipeline: "edspdf.pipeline.Pipeline",
        path: Sequence[str] = (),
    ):
        """
        We need to support passing in the pipeline object and name to factories from
        a config file. Since components can be nested, we need to add them to every
        factory in the config.
        """
        if isinstance(obj, CurriedFactory):
            if obj.error is not None:
                raise obj.error from None

            if obj.instantiated is not None:
                return obj.instantiated

            name = ".".join(path)

            kwargs = {
                key: CurriedFactory.instantiate(value, pipeline, (*path, key))
                for key, value in obj.kwargs.items()
            }
            try:
                obj.instantiated = obj.factory(
                    **{
                        "pipeline": pipeline,
                        "name": name,
                        **kwargs,
                    }
                )
            except ConfitValidationError as e:
                obj.error = ConfitValidationError(
                    patch_errors(e.raw_errors, path, model=e.model),
                    model=e.model,
                    name=obj.factory.__module__ + "." + obj.factory.__qualname__,
                ).with_traceback(None)
                raise obj.error from None
            # except Exception as e:  # pragma: no cover
            #     obj.error = ConfitValidationError([ErrorWrapper(e, path)])
            #     raise obj.error from None
            return obj.instantiated
        elif isinstance(obj, dict):
            instantiated = {}
            errors = []
            for key, value in obj.items():
                try:
                    instantiated[key] = CurriedFactory.instantiate(
                        value, pipeline, (*path, key)
                    )
                except KeyboardInterrupt:  # pragma: no cover
                    raise
                except ConfitValidationError as e:
                    errors.extend(e.raw_errors)
            if not errors:
                return instantiated
        elif isinstance(obj, (list, tuple)):
            instantiated = []
            errors = []
            for i, value in enumerate(obj):
                try:
                    instantiated.append(
                        CurriedFactory.instantiate(value, pipeline, (*path, str(i)))
                    )
                except KeyboardInterrupt:  # pragma: no cover
                    raise
                except ConfitValidationError as e:
                    errors.append(e.raw_errors)
            if not errors:
                return type(obj)(instantiated)
        else:
            return obj
        raise ConfitValidationError(list(dict.fromkeys(errors)))


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

            def invoke(validated_fn, kwargs):
                if default_config is not None:
                    kwargs = (
                        Config(default_config)
                        .resolve(registry=self.registry)
                        .merge(kwargs)
                    )
                instantiated = validated_fn(kwargs)
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
    accelerator = Registry(("edspdf", "accelerator"), entry_points=True)
    readers = Registry(("edspdf", "readers"), entry_points=True)
    writers = Registry(("edspdf", "writers"), entry_points=True)
