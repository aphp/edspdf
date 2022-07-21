# Classifiers

We developed EDS-PDF with modularity in mind. To that end, you can choose between multiple classification methods.

| Method       | Description                                                             |
| ------------ | ----------------------------------------------------------------------- |
| `mask.v1`    | Simple rule-based classification                                        |
| `sklearn.v1` | Machine-learning-base classification using a Scikit-learn pipeline      |
| `dummy.v1`   | Dummy classifier, for testing purposes. Classifies every bloc to `body` |
| `random.v1`  | To sow chaos                                                            |
