# Rule-based Extraction

Let's create a rule-based extractor for PDF documents.

!!! note

    This pipeline will likely perform poorly as soon as your PDF documents
    come in varied forms. In that case, even a very simple trained pipeline
    may give you a substantial performance boost (see [next section](sklearn.md)).

First, download this example [PDF](https://github.com/aphp/edspdf/raw/master/tests/resources/letter.pdf).

We will use the following configuration:

```toml title="config.cfg"
[reader]
@readers = "pdf-reader.v1"  # (1)

[reader.extractor]
@extractors = "pdfminer.v1"  # (2)

[reader.classifier]
@classifiers = "mask.v1"  # (3)
x0 = 0.2
x1 = 0.9
y0 = 0.3
y1 = 0.6
threshold = 0.1

[reader.aggregator]
@aggregators = "styled.v1"  # (4)
```

1. This is the top-level object, which organises the entire extraction process.
2. Here we use the provided text-based extractor, based on the PDFMiner library
3. This is where we define the rule-based classifier. Here, we use a "mask",
   meaning that every text bloc that falls within the boundaries will be assigned
   the `body` label, everything else will be tagged as pollution.
4. This aggregator returns a tuple of dictionaries. The first contains compiled text for each
   label, the second exports their style.

Save the configuration as `config.cfg` and run the following snippet:

```python
import edspdf
from pathlib import Path

reader = edspdf.load("config.cfg")  # (1)

# Get a PDF
pdf = Path("letter.pdf").read_bytes()

texts, styles = reader(pdf)
```

This code will output the following results:

=== "Visualisation"

    ![lines](resources/lines.jpeg)

=== "Extracted Text"

    ```
    Cher Pr ABC, Cher DEF,

    Nous souhaitons remercier le CSE pour son avis favorable quant à l’accès aux données de
    l’Entrepôt de Données de Santé du projet n° XXXX.

    Nous avons bien pris connaissance des conditions requises pour cet avis favorable, c’est
    pourquoi nous nous engageons par la présente à :

    • Informer individuellement les patients concernés par la recherche, admis à l'AP-HP
    avant juillet 2017, sortis vivants, et non réadmis depuis.

    • Effectuer une demande d'autorisation à la CNIL en cas d'appariement avec d’autres
    cohortes.

    Bien cordialement,
    ```

=== "Extracted Style"

    The `start` and `end` columns refer to the character indices within the extracted text.

    | fontname       | font           | style  | size   | upright | x0     | x1     | y0     | y1     | start | end |
    | -------------- | -------------- | ------ | ------ | ------- | ------ | ------ | ------ | ------ | ----- | --- |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3389 | 0.4949 | 0.3012 | 0.3130 | 0     | 22  |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3389 | 0.8024 | 0.3488 | 0.3606 | 24    | 90  |
    | BCDHEE+Calibri | BCDHEE+Calibri | Normal | 9.9600 | True    | 0.8024 | 0.8066 | 0.3488 | 0.3606 | 90    | 91  |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.8067 | 0.9572 | 0.3488 | 0.3606 | 91    | 111 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3030 | 0.3069 | 0.3655 | 0.3773 | 112   | 113 |
    | BCDHEE+Calibri | BCDHEE+Calibri | Normal | 9.9600 | True    | 0.3069 | 0.3111 | 0.3655 | 0.3773 | 113   | 114 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3111 | 0.6476 | 0.3655 | 0.3773 | 114   | 161 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3389 | 0.9327 | 0.3893 | 0.4011 | 163   | 247 |
    | BCDHEE+Calibri | BCDHEE+Calibri | Normal | 9.9600 | True    | 0.9327 | 0.9369 | 0.3893 | 0.4011 | 247   | 248 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.9369 | 0.9572 | 0.3893 | 0.4011 | 248   | 251 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3030 | 0.6440 | 0.4060 | 0.4178 | 252   | 300 |
    | SymbolMT       | SymbolMT       | Normal | 9.9600 | True    | 0.3444 | 0.3521 | 0.4299 | 0.4418 | 302   | 303 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3746 | 0.9568 | 0.4303 | 0.4422 | 304   | 386 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3746 | 0.7544 | 0.4470 | 0.4588 | 387   | 445 |
    | SymbolMT       | SymbolMT       | Normal | 9.9600 | True    | 0.3444 | 0.3521 | 0.4710 | 0.4828 | 447   | 448 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3746 | 0.9096 | 0.4714 | 0.4832 | 449   | 523 |
    | BCDHEE+Calibri | BCDHEE+Calibri | Normal | 9.9600 | True    | 0.9097 | 0.9139 | 0.4714 | 0.4832 | 523   | 524 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.9139 | 0.9572 | 0.4714 | 0.4832 | 524   | 530 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3746 | 0.4389 | 0.4882 | 0.5000 | 531   | 540 |
    | BCDFEE+Calibri | BCDFEE+Calibri | Normal | 9.9600 | True    | 0.3389 | 0.4678 | 0.5357 | 0.5475 | 542   | 560 |
