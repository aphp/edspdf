# Box classifiers

We developed EDS-PDF with modularity in mind. To that end, you can choose between multiple classification methods.

<!-- --8<-- [start:components] -->

| Factory name                                                                                     | Description                             |
|--------------------------------------------------------------------------------------------------|-----------------------------------------|
| [`mask-classifier`][edspdf.pipes.classifiers.mask.simple_mask_classifier_factory]           | Simple rule-based classification        |
| [`multi-mask-classifier`][edspdf.pipes.classifiers.mask.mask_classifier_factory]            | Simple rule-based classification        |
| [`dummy-classifier`][edspdf.pipes.classifiers.dummy.DummyClassifier]                        | Dummy classifier, for testing purposes. |
| [`random-classifier`][edspdf.pipes.classifiers.random.RandomClassifier]                     | To sow chaos                            |
| [`trainable-classifier`][edspdf.pipes.classifiers.trainable.TrainableClassifier] | Trainable box classification model      |

<!-- --8<-- [end:components] -->
