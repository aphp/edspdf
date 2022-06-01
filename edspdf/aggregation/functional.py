import pandas as pd


def prepare_newlines(
    lines: pd.DataFrame, nl_threshold: float, np_threshold
) -> pd.DataFrame:

    # Get information
    lines["next_y1"] = lines.groupby(["label"])["y1"].shift(-1)
    lines["next_page"] = lines.groupby(["label"])["page"].shift(-1)

    lines["dy"] = (lines.next_y1 - lines.y1).where(lines.next_y1 > lines.y1)
    lines["med_dy"] = lines.groupby(["label"])["dy"].transform("median")

    lines["newline"] = " "

    lines.newline = lines.newline.mask(
        lines.dy > lines.med_dy * nl_threshold,
        "\n",
    )
    lines.newline = lines.newline.mask(
        lines.dy > lines.med_dy * np_threshold,
        "\n\n",
    )

    lines.newline = lines.newline.mask(
        lines.page != lines.next_page,
        "\n\n",
    )

    lines["text_with_newline"] = lines.text + lines.newline

    return lines
