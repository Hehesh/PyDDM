import numpy as np
import pandas as pd
from fixations import derasterize_fixations

def derasterize_data(
    df: pd.DataFrame,
    trial_col: str,
    start_col: str = "fix_start",
    end_col: str = "fix_end",
    loc_col: str = "fix_location",
    fill_code: int = 0,
    dtype=np.int8,
    seq_col: str = "fix_sequence",
) -> pd.DataFrame:
    """
    Transform fixation data from start/end timestamps into per-trial fixation sequences.

    Parameters
    -------
    df : pd.DataFrame
        Long-format DataFrame containing fixation data with at least the columns
        specified by `trial_col`, `start_col`, `end_col`, and `loc_col`.
    trial_col : str
        Name of the column identifying each trial.
    start_col : str, default "fix_start"
        Column containing the inclusive start timestamp of each fixation.
    end_col : str, default "fix_end"
        Column containing the inclusive end timestamp of each fixation.
    loc_col : str, default "fix_location"
        Column containing integer codes for fixation locations.
    fill_code : int, default 0
        Value used to fill time indices not covered by any fixation (gaps).
    dtype : np.dtype, default np.int8
        Data type of the resulting fixation sequences.
    seq_col : str, default "fix_sequence"
        Name of the new column containing each trial’s fixation sequence array.

    Outputs
    --------
    pd.DataFrame
        A trial-level DataFrame where each row corresponds to a single trial.
        Columns include:
        - The preserved metadata columns from the first fixation of each trial.
        - A `seq_col` column containing a NumPy array (1D) representing the
          full fixation sequence for that trial.
    """
    # stable sort so equal starts keep input order
    df_sorted = df.sort_values([trial_col, start_col], kind="mergesort")

    rows = []
    for _, g in df_sorted.groupby(trial_col, sort=False):
        # build sequence for this trial
        seq = derasterize_fixations(
            g[[start_col, end_col, loc_col]],
            start_col=start_col,
            end_col=end_col,
            loc_col=loc_col,
            fill_code=fill_code,
            dtype=dtype,
        )

        # take the first row's metadata
        first = g.iloc[0]
        cols = list(dict.fromkeys(([trial_col, loc_col])))
        data = {c: first[c] for c in cols if c in g.columns}

        data[seq_col] = seq
        rows.append(data)

    return pd.DataFrame(rows)
