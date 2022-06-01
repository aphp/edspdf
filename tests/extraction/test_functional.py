from email.policy import strict

import pandas as pd

from edspdf.extraction.functional import remove_outside_lines


def test_removal():

    lines = pd.DataFrame.from_records(
        [
            dict(x0=0.1, x1=0.9, y0=0.1, y1=0.9),
            dict(x0=-0.2, x1=0.2, y0=0.1, y1=0.9),
            dict(x0=-0.9, x1=-0.1, y0=0.1, y1=0.9),
        ]
    )

    assert len(remove_outside_lines(lines)) == 2
    assert len(remove_outside_lines(lines, strict_mode=True)) == 1
