# Changelog

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
