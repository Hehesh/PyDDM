import numpy as np
import pandas as pd
from pyddm.preprocessing.fixations import *

# tabular to long form
def rasterize_data(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject",
    trial_col: str = "trial",
    seq_col: str = "sequence",
    keep_cols: "str | list[str]" = "all",
    process_cols: bool = False,
    drop_seq_in_output: bool = True,
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
    req = [subject_col, trial_col, seq_col]
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
        for c in (subject_col, trial_col):
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

    if process_cols:
        # Calculate fixation duration, order, and reverse order
        out_df = pd.concat(out_frames, ignore_index=True)
        out_df['fix_duration'] = out_df['fix_end'] - out_df['fix_start']
        out_df['fix_num'] = out_df.groupby([subject_col, trial_col]).cumcount() + 1
        out_df['fix_num_rev'] = out_df.groupby([subject_col, trial_col])['fix_num'].transform(
            lambda x: x.max() - x + 1
        )
        return out_df

    return pd.concat(out_frames, ignore_index=True)

# long form to tabular
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
    process_cols: list[str] | None = None,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Construct per-(subject, trial) fixation sequences from raw fixation data,
    while preserving trial-level columns.

    Parameters
    ----------
    process_cols : list[str] or None
        Names for processed columns as a result of rasterize data. If None,
        no additional columns are excluded.
    keep_cols : list[str] or None
        Additional columns to carry through to the output. These must be
        constant within each (subject, trial). If None, automatically keeps
        all non-fixation columns.
    """

    if df[subject_col].isna().any() or df[trial_col].isna().any():
        raise ValueError(f"{subject_col} and {trial_col} must not contain NaNs.")

    df = df.copy()

    # Columns used for fixation structure
    fixation_cols = {start_col, end_col, loc_col}  
    if process_cols is not None:
        fixation_cols.update(process_cols)

    # Determine columns to carry forward
    if keep_cols is None:
        keep_cols = [
            c for c in df.columns
            if c not in fixation_cols and c not in {subject_col, trial_col}
        ]

    df_sorted = df.sort_values([subject_col, trial_col, start_col], kind="mergesort")

    rows = []

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
            trial_col: int(first[trial_col])
        }

        # Carry trial-level columns
        for col in keep_cols:
            data[col] = first[col]

        data[seq_col] = seq

        rows.append(data)

    result = pd.DataFrame(rows)

    # Ensure correct dtypes
    result[subject_col] = result[subject_col].astype(int)
    result[trial_col] = result[trial_col].astype(int)

    return result