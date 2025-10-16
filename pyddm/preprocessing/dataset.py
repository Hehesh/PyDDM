import numpy as np
import pandas as pd
from pyddm.preprocessing.fixations import derasterize_fixations

def derasterize_data(
    df: pd.DataFrame,
    subject_col: str,
    trial_col: str,
    start_col: str = "fix_start",
    end_col: str = "fix_end",
    loc_col: str = "fix_location",
    fill_code: int = 0,
    dtype=np.int8,
    seq_col: str = "fix_sequence",
) -> pd.DataFrame:
    """
    Construct per-(subject, trial) fixation sequences from raw fixation data.

    This function aggregates fixation-level data into per-trial sequences of
    fixation locations for each subject. Each (subject, trial) group is
    derasterized into a time-indexed NumPy array using
    `derasterize_fixations()`.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame containing fixation-level data with at least
        the columns specified by `subject_col`, `trial_col`, `start_col`,
        `end_col`, and `loc_col`.
    subject_col : str
        Column identifying each subject or participant.
    trial_col : str
        Column identifying the trial number within each subject.
    start_col : str, default "fix_start"
        Column containing the inclusive start timestamp of each fixation.
    end_col : str, default "fix_end"
        Column containing the inclusive end timestamp of each fixation.
    loc_col : str, default "fix_location"
        Column containing integer-coded fixation locations.
    fill_code : int, default 0
        Integer code used to fill time indices not covered by a fixation.
    dtype : np.dtype, default np.int8
        Data type of the resulting fixation sequences.
    seq_col : str, default "fix_sequence"
        Name of the new column containing the per-trial fixation sequence.

    Returns
    -------
    result : pd.DataFrame
        A trial-level DataFrame with one row per (subject, trial) pair.
        Columns include:
        - `subject_col` (int): subject ID
        - `trial_col` (int): trial ID
        - `loc_col` (int): location code from the first fixation
        - `seq_col` (np.ndarray): 1D NumPy array (dtype=`dtype`) representing
          the full derasterized fixation sequence.
    """

    if df[subject_col].isna().any() or df[trial_col].isna().any():
        raise ValueError(f"{subject_col} and {trial_col} must not contain NaNs.")

    df = df.copy()

    df_sorted = df.sort_values([subject_col, trial_col, start_col], kind="mergesort")

    rows = []
    seqs = []

    for _, g in df_sorted.groupby([subject_col, trial_col], sort=False):
        seq = derasterize_fixations(
            g,
            start_col=start_col,
            end_col=end_col,
            loc_col=loc_col,
            fill_code=fill_code,
            dtype=dtype,
        )

        first = g.iloc[0]
        data = {
            subject_col: int(first[subject_col]),
            trial_col: int(first[trial_col]),
            loc_col: int(first[loc_col]),
        }

        rows.append(data)
        seqs.append(np.array(seq, dtype=dtype, copy=True))

    result = pd.DataFrame(rows)
    result[subject_col] = result[subject_col].astype(int)
    result[trial_col] = result[trial_col].astype(int)
    result[loc_col] = result[loc_col].astype(int)
    result[seq_col] = pd.Series(seqs, dtype=object)

    return result