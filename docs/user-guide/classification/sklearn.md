# Sklearn

We provide a helper to use a Scikit-Learn pipeline for the classification task.

Save your pipeline with joblib (as advised in the [Scikit-Learn documentation](https://scikit-learn.org/stable/modules/model_persistence.html#model-persistence)),
and use the helper registered as `sklearn.v1`. It expects a path to your saved configuration.

```conf
[classifier]
@classifiers = "sklearn.v1"
path = "pipeline.joblib"
```
