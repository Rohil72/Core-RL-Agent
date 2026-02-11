import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Persist a dataframe with parquet preferred and pickle fallback.

    When parquet dependencies are unavailable, this writes a pickle payload to
    the same path so callers can keep their existing file naming conventions.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(out)
        return
    except Exception as exc:
        logger.warning("Parquet write failed for %s; falling back to pickle: %s", out, exc)

    df.to_pickle(out)


def read_dataframe(path: str) -> pd.DataFrame:
    """
    Load a dataframe with parquet preferred and pickle fallback.
    """
    p = Path(path)

    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.warning("Parquet read failed for %s; trying pickle fallback: %s", p, exc)

    return pd.read_pickle(p)
