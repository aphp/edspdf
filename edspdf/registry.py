from typing import Callable, Optional

import catalogue


class SubRegistry(catalogue.Registry):
    def register(
        self, name: str, *, func: Optional[catalogue.InFunc] = None
    ) -> Callable[[catalogue.InFunc], catalogue.InFunc]:
        from .config import validate_arguments

        registerer = super().register(name)

        def wrap_and_register(fn: catalogue.InFunc) -> catalogue.InFunc:
            fn = validate_arguments(
                func=fn,
                config={"arbitrary_types_allowed": True},
                save_params={f"@{self.namespace[-1]}": name},
            )
            return registerer(fn)

        if func is not None:
            return wrap_and_register(func)
        else:
            return wrap_and_register


class Registry:
    factory = SubRegistry(("edspdf", "factory"), entry_points=True)
    adapter = SubRegistry(("edspdf", "adapter"), entry_points=True)
    misc = SubRegistry(("edspdf", "misc"), entry_points=True)

    _catalogue = dict(
        factory=factory,
        adapter=adapter,
        misc=misc,
    )

    def resolve(self, config):
        from .config import Config

        return Config(__root__=config).resolve(deep=True)["__root__"]


registry = Registry()
