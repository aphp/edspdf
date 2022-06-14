# Changelog

## v0.4.1 - 2022-06-14

### Fixed

- Remove `importlib.metadata` dependency, which led to issues with Python 3.7

## v0.4.0 - 2022-06-14

### Added

- Python 3.7 support, by relaxing dependency constraints
- Support for package-resource pipeline for `sklearn-pipeline.v1`

## v0.3.2 - 2022-06-03

### Added

- `compare_results` in visualisation

## v0.3.1 - 2022-06-02

### Fixed

- Rescale transform now keeps origin on top-left corner

## v0.3.0 - 2022-06-01

### Added

- Styles management within the extractor
- `styled.v1` aggregator, to handle styles
- `rescale.v1` transform, to go back to the original height and width

### Changed

- Styles and text extraction is handled by the extractor directly
- The PDFMiner `line` object is not carried around any more

### Removed

- Outdated `params` entry in the EDS-PDF registry.

## v0.2.2 - 2022-05-12

### Changed

- Fixed `merge_lines` bug when lines were empty
- Modified the demo consequently

## v0.2.1 - 2022-05-09

### Changed

- The extractor **always** returns a pandas DataFrame, be it empty. It enhances robustness and stability.

## v0.2.0 - 2022-05-09

### Added

- `aggregation` submodule to handle the specifics of aggregating text blocs
- Base classes for better-defined modules
- Uniformise the columns to `labels`
- Add arbitrary contextual information

### Removed

- `typer` legacy dependency
- `models` submodule, which handled the configurations for Spark distribution (deferred to another package)
- specific `orbis` context, which was APHP-specific

## v0.1.0 - 2022-05-06

Inception ! :tada:

### Features

- spaCy-like configuration system
- Available classifiers :
  - `dummy.v1`, that classifies everything to `body`
  - `mask.v1`, for simple rule-based classification
  - `sklearn.v1`, that uses a Scikit-Learn pipeline
  - `random.v1`, to better sow chaos
- Merge different blocs together for easier visualisation
- Streamlit demo with visualisation
