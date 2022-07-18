# Mask Classification

We developed a simple classifier that roughly uses the same strategy as PDFBox, namely:

- define a "mask" on the PDF documents ;
- keep every text bloc within that mask, tag everything else as pollution.

## Factories

Two factories are available in the `classifiers` registry: `mask.v1` and `custom_masks.v1`.

### `mask.v1`

The simplest form. You define the mask, everything else is tagged as a pollution.

Example configuration :

```toml
[classifier]
@classifiers = "mask.v1"
x0 = 0.1
y0 = 0.1
x1 = 0.9
y1 = 0.9
threshold = 0.9
```

### `custom_masks.v1`

A generalisation, wherein the user defines a number of regions.

The following configuration produces _exactly_ the same classifier as `mask.v1` example above.

```toml
[classifier]
@classifiers = "custom_masks.v1"

[classifier.body]
label = "body"
x0 = 0.1
y0 = 0.1
x1 = 0.9
y1 = 0.9
threshold = 0.9
```

The following configuration defines a `header` region.

```toml
[classifier]
@classifiers = "custom_masks.v1"

[classifier.header]
label = "header"
x0 = 0.1
y0 = 0.1
x1 = 0.9
y1 = 0.3
threshold = 0.9

[classifier.body]
label = "body"
x0 = 0.1
y0 = 0.3
x1 = 0.9
y1 = 0.9
threshold = 0.9
```

Any bloc that is not part of a mask is tagged as `pollution`.
