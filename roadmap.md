# EDS-PDF Roadmap

The goal of EDS-PDF is to provide an efficient and customisable way of extracting text from PDFs.

To remain as modular as possible, we break the task into self contained subtasks.

| Subtask                     | Description                                                                  | Input     | Output    |
| --------------------------- | ---------------------------------------------------------------------------- | --------- | --------- |
| Extract lines               | Extract rich representation (text, position, style) of text blocs from a PDF | PDF bytes | dataframe |
| Transformation/Augmentation | Add relevant information before classification                               | dataframe | dataframe |
| Classification              | Classify each line                                                           | dataframe | dataframe |
| Aggregation                 | Merge texts together, possibly handling tables                               | dataframe | dataframe |

We can use EDS-NLP to extract relevant administrative information.
