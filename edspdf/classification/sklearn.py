from pathlib import Path

from joblib import load

from edspdf.utils.registry import registry


@registry.classifiers.register("sklearn-pipeline.v1")
def sklearn_pipeline_factory(
    path: Path,
):
    return load(path)
