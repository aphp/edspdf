import numpy as np
import pandas as pd


def prepare_newlines(
    lines: pd.DataFrame, nl_threshold: float, np_threshold
) -> pd.DataFrame:

    # Get information
    lines["next_y1"] = lines.groupby(["label"])["y1"].shift(-1)
    lines["next_page"] = lines.groupby(["label"])["page"].shift(-1)

    lines["dy"] = (lines.next_y1 - lines.y1).where(lines.next_y1 > lines.y1)
    lines["height"] = lines.y1 - lines.y0
    lines["next_height"] = lines.groupby(["label"])["height"].shift(-1)
    lines["min_height"] = np.minimum(lines["height"], lines["next_height"])

    lines["newline"] = " "

    lines.newline = lines.newline.mask(
        lines.dy > lines.min_height * nl_threshold,
        "\n",
    )
    lines.newline = lines.newline.mask(
        lines.dy > lines.min_height * np_threshold,
        "\n\n",
    )

    lines.newline = lines.newline.mask(
        lines.page != lines.next_page,
        "\n\n",
    )

    lines["text_with_newline"] = lines.text + lines.newline

    return lines
