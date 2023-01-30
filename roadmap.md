# Roadmap

- [x] Style extraction
- [x] Custom hybrid torch-based pipeline & configuration system
- [x] Drop pandas DataFrame in favour of a ~~Cython~~ [attr](https://www.attrs.org/en/stable/) wrapper around PDF documents?
- [x] Add training capabilities with a CLI to automate the annotation/preparation/training loop.
      Again, draw inspiration from spaCy, and maybe add the notion of a `TrainableClassifier`...
- [ ] Add complete serialisation capabilities, to save a full pipeline to disk.
      Draw inspiration from spaCy, which took great care to solve these issues:
      add `save` and `load` methods to every pipeline component
- [ ] Multiple-column extraction
- [ ] Table detector
- [ ] Integrate third-party OCR module
