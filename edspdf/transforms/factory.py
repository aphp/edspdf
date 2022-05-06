from edspdf.utils.registry import registry

from .transforms import (
    ChainTransform,
    add_dates,
    add_dimensions,
    add_orbis,
    add_telephone,
)


@registry.transforms.register("chain.v1")
def chain_transforms_factory(
    *layers,
):
    chained = ChainTransform(*layers)
    return chained


@registry.transforms.register("telephone.v1")
def telephone_transform_factory():
    return add_telephone


@registry.transforms.register("dates.v1")
def dates_transform_factory():
    return add_dates


@registry.transforms.register("orbis.v1")
def orbis_transform_factory():
    return add_orbis


@registry.transforms.register("dimensions.v1")
def dimensions_transform_factory():
    return add_dimensions
