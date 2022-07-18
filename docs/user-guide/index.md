# Overview

EDS-PDF is organised into submodules that divide the text extraction process between well-defined steps:

1. Extract the text blocs from the PDF. For now, EDS-PDF only handles text-based PDF, but OCR could be added in the future.
   The output of this step is a pandas DataFrame.
2. Transform the DataFrame to add rule-based information useful for the classification step
3. Classify text blocs (typically between `body`, `header` and `footer`, but that's up to you)
4. Aggregate the text

## Data Structure

EDS-PDF parses the PDF into a Pandas DataFrame object where each row represents a text bloc. The dataframe is carried all the way to the aggregation step.

The following columns are reserved:

| Column        | Description                                |
| ------------- | ------------------------------------------ |
| `text`        | Bloc text content                          |
| `page`        | Page within the PDF (starting at 0)        |
| `x0/y0/x1/y1` | Bloc bounding box (from top left corner)   |
| `label`       | Class of the bloc (eg `body`, `header`...) |

## Utilities

EDS-PDF also provides a number of utilities, for instance to merge same-label zones together for better usability.
