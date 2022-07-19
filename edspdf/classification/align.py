import pandas as pd


def align_labels(
    lines: pd.DataFrame,
    labels: pd.DataFrame,
    threshold: float = 0.0001,
) -> pd.DataFrame:
    """
    Align lines with possibly overlapping (and non-exhaustive) labels.

    Possible matches are sorted by covered area. Lines with no overlap at all

    Parameters
    ----------
    lines : pd.DataFrame
        DataFrame containing the lines
    labels : pd.DataFrame
        DataFrame containing the labels
    threshold : float, default 1
        Threshold to use for discounting a label. Used if the `labels` DataFrame
        does not provide a `threshold` column, or to fill `NaN` values thereof.

    Returns
    -------
    pd.DataFrame
        A copy of the lines table, with the labels added.
    """

    lines["uid"] = range(len(lines))

    df = lines[
        sorted({"uid", "page", "x0", "y0", "x1", "y1"} & set(lines.columns))
    ].copy()
    labels = labels.copy()

    if "threshold" not in labels.columns:
        labels["threshold"] = threshold

    labels.threshold = labels.threshold.fillna(threshold)

    df = df.merge(
        labels, how="inner" if set(df.columns) & set(labels.columns) else "cross"
    )

    df["dx"] = df[["x1", "X1"]].min(axis=1) - df[["x0", "X0"]].max(axis=1)
    df["dy"] = df[["y1", "Y1"]].min(axis=1) - df[["y0", "Y0"]].max(axis=1)

    df["overlap"] = (df.dx > 0) * (df.dy > 0) * df.dx * df.dy

    df["area"] = (df.x1 - df.x0) * (df.y1 - df.y0)
    df["ratio"] = df.overlap / df.area

    df["area_mask"] = (df.X1 - df.X0) * (df.Y1 - df.Y0)
    df["ratio_mask"] = df.overlap / df.area_mask

    df["thresholded"] = df.ratio >= df.threshold

    df = df.sort_values(["thresholded", "ratio_mask"], ascending=False)

    df = df.groupby(["uid"], as_index=False).first()
    df = df.sort_values("uid").reset_index(drop=True)

    df.label = df.label.where(df.thresholded)

    df = lines.merge(df[["uid", "label"]], on="uid").drop(columns=["uid"])
    lines.drop(columns="uid", inplace=True)

    return df
