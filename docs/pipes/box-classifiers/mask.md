# Mask Classification

We developed a simple classifier that roughly uses the same strategy as PDFBox, namely:

- define a "mask" on the PDF documents ;
- keep every text bloc within that mask, tag everything else as pollution.

## Factories

Two factories are available in the `classifiers` registry: `mask-classifier` and `multi-mask-classifier`.

### `mask-classifier` {: #edspdf.pipes.classifiers.mask.simple_mask_classifier_factory }

::: edspdf.pipes.classifiers.mask.simple_mask_classifier_factory
    options:
        heading_level: 4
        show_bases: false
        show_source: false

---

### `multi-mask-classifier` {: #edspdf.pipes.classifiers.mask.mask_classifier_factory }

::: edspdf.pipes.classifiers.mask.mask_classifier_factory
    options:
        heading_level: 4
        show_bases: false
        show_source: false
