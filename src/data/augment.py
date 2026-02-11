import numpy as np
import pandas as pd


def augment_3day_geometry(
    df: pd.DataFrame,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """
    Convert daily OHLCV into centered 3-day geometric features.

    Input df columns:
        open, high, low, close, volume
    Index:
        DatetimeIndex (daily)

    Output columns:
        ret_3d
        range_3d
        close_pos_3d
        vol_ratio_3d
    """

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    df = df.sort_index()

    records = []

    for i in range(1, len(df) - 1):
        window = df.iloc[i - 1 : i + 2]

        # skip incomplete windows
        if window.isnull().any().any():
            continue

        o_start = window["open"].iloc[0]
        c_end = window["close"].iloc[-1]
        h_max = window["high"].max()
        l_min = window["low"].min()

        # log return
        ret = np.log(c_end / o_start)

        # range ratio
        range_ratio = (h_max - l_min) / max(o_start, 1e-8)

        # close position in range
        close_pos = (c_end - l_min) / max(h_max - l_min, 1e-8)

        # volume normalization (rolling baseline)
        if i < vol_lookback:
            continue  # insufficient volume context

        vol_window = window["volume"].mean()
        vol_baseline = df["volume"].iloc[i - vol_lookback : i].mean()
        vol_ratio = np.log(vol_window / max(vol_baseline, 1e-8))

        records.append(
            {
                "date": df.index[i],
                "ret_3d": ret,
                "range_3d": range_ratio,
                "close_pos_3d": close_pos,
                "vol_ratio_3d": vol_ratio,
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("date")
        .sort_index()
    )
