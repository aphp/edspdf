from typing import Any, Dict

import catalogue


class Registry:

    extractors = catalogue.create("edspdf", "extractors", entry_points=True)
    readers = catalogue.create("edspdf", "readers", entry_points=True)
    aggregators = catalogue.create("edspdf", "aggregators", entry_points=True)
    transforms = catalogue.create("edspdf", "transforms", entry_points=True)
    classifiers = catalogue.create("edspdf", "classifiers", entry_points=True)
    misc = catalogue.create("edspdf", "misc", entry_points=True)

    _catalogue = dict(
        extractors=extractors,
        aggregators=aggregators,
        readers=readers,
        classifiers=classifiers,
        transforms=transforms,
        misc=misc,
    )

    def resolve(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:

        filled = dict()

        for key, value in config.items():
            if isinstance(value, dict):
                filled[key] = self.resolve(value)
            else:
                filled[key] = value

        first_key = list(filled.keys())[0]

        if first_key.startswith("@"):

            reg = self._catalogue[first_key[1:]]

            # Handle list of arguments
            args = filled.pop("*", dict()).values()
            return reg.get(filled.pop(first_key))(*args, **filled)

        return filled


registry = Registry()
