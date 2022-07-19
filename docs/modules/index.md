# Overview

EDS-PDF is organised into modules that handle each step of the extraction process.

The following functions are registered:

| Section       | Function              |
| ------------- | --------------------- |
| `extractors`  | `pdfminer.v1`         |
| `aggregators` | `simple.v1`           |
| `aggregators` | `styled.v1`           |
| `readers`     | `pdf-reader.v1`       |
| `classifiers` | `dummy.v1`            |
| `classifiers` | `mask.v1`             |
| `classifiers` | `custom_masks.v1`     |
| `classifiers` | `random.v1`           |
| `classifiers` | `sklearn-pipeline.v1` |
| `transforms`  | `chain.v1`            |
| `transforms`  | `telephone.v1`        |
| `transforms`  | `dates.v1`            |
| `transforms`  | `dimensions.v1`       |
| `transforms`  | `rescale.v1`          |
| `misc`        | `package-resource.v1` |
