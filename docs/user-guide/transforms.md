# Transforms

The transformation phase consists of rule-based transformations applied to the text blocs
in order to provide the classification algorithm with more information.

| Method          | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| `chain.v1`      | Chain a sequence of transforms together                         |
| `telephone.v1`  | Creates a new column, containing the number of phone numbers    |
| `dates.v1`      | Creates a new column, containing the number of dates            |
| `dimensions.v1` | Creates new columns with the width, height and area of the bloc |
| `rescale.v1`    | Rescale the bloc dimensions to the original height and width    |
