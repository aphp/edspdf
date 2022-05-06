from typing import Any, Dict

import catalogue


class Registry(object):

    extractors = catalogue.create("edspdf", "extractors")
    params = catalogue.create("edspdf", "params")
    readers = catalogue.create("edspdf", "readers")
    transforms = catalogue.create("edspdf", "transforms")
    classifiers = catalogue.create("edspdf", "classifiers")

    _catalogue = dict(
        extractors=extractors,
        params=params,
        readers=readers,
        classifiers=classifiers,
        transforms=transforms,
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
