import numpy as np
import pandas as pd
from pyddm.preprocessing.fixations import *

def rasterize_data(
    df: pd.DataFrame,
    *,
    parcode_col: str = "parcode",
    trial_col: str = "trial",
    seq_col: str = "sequence",
    keep_cols: "str | list[str]" = "all",   # "all" or a list of column names to keep
    drop_seq_in_output: bool = True,        # don't include the sequence column in the output
) -> pd.DataFrame:
    """
    Expand per-row fixation sequences into long-format fixations and
    preserve original row metadata.

    Returns a DataFrame with:
      [<kept metadata cols>, 'fix_start', 'fix_end', 'fix_location']

    Parameters
    ----------
    df : DataFrame with at least [parcode_col, trial_col, seq_col]
    parcode_col, trial_col, seq_col : str
        Column names for participant, trial, and 1D fixation sequence.
    keep_cols : "all" | list[str]
        - "all": keep every column from `df` except `seq_col` (default).
        - list: keep only these columns (they will be added to the output).
    drop_seq_in_output : bool
        If True, the sequence column is not included in the output.
    """
    req = [parcode_col, trial_col, seq_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Decide which metadata columns to keep
    if keep_cols == "all":
        meta_cols = [c for c in df.columns if c != seq_col]
        if drop_seq_in_output is False:  # rare: user wants sequence copied too
            meta_cols = list(df.columns)
    else:
        meta_cols = list(dict.fromkeys(keep_cols))  # dedupe, preserve order
        # Always ensure IDs exist in output
        for c in (parcode_col, trial_col):
            if c not in meta_cols:
                meta_cols.append(c)

    out_frames = []

    for _, row in df.iterrows():
        seq = np.asarray(row[seq_col])
        ras = rasterize_fixations(seq)  # uses your inclusive convention, skips 0s
        if ras.empty:
            continue

        # Attach metadata from the source row to each rasterized segment
        for c in meta_cols:
            ras[c] = row[c]

        # Order columns: metadata first, then fixation columns
        cols_order = [*meta_cols, "fix_start", "fix_end", "fix_location"]
        ras = ras[cols_order]
        out_frames.append(ras)

    if not out_frames:
        return pd.DataFrame(columns=[*meta_cols, "fix_start", "fix_end", "fix_location"])

    return pd.concat(out_frames, ignore_index=True)

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

    count = 0

    for _, g in df_sorted.groupby([subject_col, trial_col], sort=False):
        seq = derasterize_fixations(
            g,
            start_col=start_col,
            end_col=end_col,
            loc_col=loc_col,
            fill_code=fill_code,
            dtype=dtype,
        )
        if count == 0:
            print(g)
            count += 1

        first = g.iloc[0]
        data = {
            subject_col: int(first[subject_col]),
            trial_col: int(first[trial_col]),
            loc_col: int(first[loc_col]),
            seq_col: np.array(seq, dtype=dtype, copy=True)
        }

        rows.append(data)

    result = pd.DataFrame(rows)
    result[subject_col] = result[subject_col].astype(int)
    result[trial_col] = result[trial_col].astype(int)
    result[loc_col] = result[loc_col].astype(int)

    return result