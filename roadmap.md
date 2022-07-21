# Roadmap

- [x] Style extraction
- [ ] spaCy classifier, to use richer text representations
- [ ] Add complete serialisation capabilities, to save a full pipeline to disk.
      Draw inspiration from spaCy, which took great care to solve these issues:
      add `save` and `load` methods to every pipeline component
- [ ] Add training capabilities with a CLI to automate the annotation/preparation/training loop.
      Again, draw inspiration from spaCy, and maybe add the notion of a `TrainableClassifier`...
- [ ] Multiple-column extraction
- [ ] Table detector
- [ ] Integrate third-party OCR module
