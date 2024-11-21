# Changelog

## v0.9.3

- Support pydantic v2

## v0.9.2

### Changed

- Default to fp16 when inferring with gpu
- Support `inputs` parameter in `TrainablePipe.postprocess(...)` method (as in edsnlp)
- We now check that the user isn't trying to write a single file in a split fashion (when `write_in_worker is True ` or `num_rows_per_file is not None`) and raise an error if they do

### Fixed

- Batches full of empty content boxes no longer crash the `huggingface-embedding` component
- Ensure models are always loaded in non training mode
- Improved performance of `edspdf.data` methods over a filesystem (`fs` parameter)

## v0.9.1

### Fixed

- It is now possible to recursively retrieve pdf files in a directory using `edspdf.data.read_files`

## v0.9.0

### Added

- New unified `edspdf.data` api (pdf files, pandas, parquet) and LazyCollection object
  to efficiently read / write data from / to different formats & sources. This API is
  has been heavily inspired by the `edsnlp.data` API.
- New unified processing API to select the execution backend via `data.set_processing(...)`
  to replace the old `accelerators` API (which is now deprecated, but still available).
- `huggingface-embedding` now supports quantization and other `AutoModel.from_pretrained` kwargs
- It is now possible to add convert a label to multiple labels in the `simple-aggregator` component :

```ini
# To build the "text" field, we will aggregate "title", "body" and "table" lines,
# and output "title" lines in a separate field as well.
label_map = {
    "text" : [ "title", "body", "table" ],
    "title": "title",
    }
```

### Fixed

- `huggingface-embedding` now resize bbox features for large PDFs, instead of making the model crash
- `huggingface-embedding` and `sub-box-cnn-pooler` now handle empty PDFs correctly

## v0.8.1

### Fixed

- Fix typing to allow passing an accelerator dict to `Pipeline.pipe(...)`
- Removed multiprocessing accelerator debug output
- Fixed absolute links in github-pages docs (e.g. image assets)

### Changed

- Added auto-links to components in the docs (by comparing span contents with entry points)

## v0.8.0

### Added

- Add multi-modal transformers (`huggingface-embedding`) with windowing options
- Add `render_page` option to `pdfminer` extractor, for multi-modal PDF features
- Add inference utilities (`accelerators`), with simple mono process support and multi gpu / cpu support
- Packaging utils (`pipeline.package(...)`) to make a pip installable package from a pipeline

### Changed

- Updated API to follow EDS-NLP's refactoring
- Updated `confit` to 0.4.2 (better errors) and `foldedtensor` to 0.3.0 (better multiprocess support)
- Removed `pipeline.score`. You should use `pipeline.pipe`, a custom scorer and `pipeline.select_pipes` instead.
- Better test coverage
- Use `hatch` instead of `setuptools` to build the package / docs and run the tests

### Fixed

- Fixed `attrs` dependency only being installed in dev mode

## v0.7.0

Major refactoring of the library:

### Core features
- new pipeline system whose API is inspired by spaCy
- first-class support for pytorch
- hybrid model inference and training (rules + deep learning)
- moved from pandas DataFrame to attrs dataclasses (`PDFDoc`, `Page`, `Box`, ...) for representing PDF documents
- new configuration system based on [config][https://github.com/aphp/config], with support for instantiation of complex deep learning models, off-the-shelf CLI, ...

### Functional features
- new extractors: pymupdf and poppler (separate packages for licensing reasons)
- many deep learning layers (box-transformer, 2d attention with relative position information, ...)
- trainable deep learning classifier
- training recipes for deep learning models

## v0.6.3 - 2023-01-23

### Fixed

- Allow corrupted PDF to not raise an error by default (they are treated as empty PDFs)
- Fix classification and aggregation for empty PDFs

## v0.6.2 - 2022-12-07

Cast bytes-like extractor inputs as bytes

## v0.6.1 - 2022-12-07

Performance and cuda related fixes.

## v0.6.0 - 2022-12-05

Many, many changes:
- added torch as the main deep learning framework instead of spaCy and thinc :tada:
- added poppler and mupdf as alternatives to pdfminer
- new pipeline / config / registry system to facilitate consistency between training and inference
- standardization of the exchange format between components with dataclass models (attrs more specifically) instead of pandas dataframes

## v0.5.3 - 2022-08-31

### Added

- Add label mapping parameter to aggregators (to merge different types of blocks such as `title` and `body`)
- Improved line aggregation formula

## v0.5.2 - 2022-08-30

### Fixed

- Fix aggregation for empty documents

## v0.5.1 - 2022-07-26

### Changed

- Drop the `pdf2image` dependency, replacing it with `pypdfium2` (easier installation)

## v0.5.0 - 2022-07-25

### Changed

- Major refactoring of the library. Moved from concepts (`aggregation`) to plural names (`aggregators`).

## v0.4.3 - 2022-07-20

### Fixed

- Multi page boxes alignment

## v0.4.2 - 2022-07-06

### Added

- `package-resource.v1` in the misc registry

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
