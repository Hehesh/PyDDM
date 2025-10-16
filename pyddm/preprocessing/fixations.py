import numpy as np
import pandas as pd

def rasterize_fixations(seq: np.ndarray) -> pd.DataFrame:
    """
    Single-trial: sequence (1D array of ints) -> long DataFrame of fixations.
    Keeps only columns ['fix_start', 'fix_end', 'fix_location'].

    Conventions
    -----------
    - fix_start and fix_end are BOTH inclusive.
    - fix_location is the code value at that fixation.
    """
    seq = np.asarray(seq)
    if seq.ndim != 1:
        raise ValueError("Input must be a 1D numpy array representing one trial.")
    if seq.size == 0:
        return pd.DataFrame(columns=["fix_start", "fix_end", "fix_location"])

    # Run-length encode
    changes = np.flatnonzero(np.diff(seq, prepend=seq[0] - 1))
    starts = changes
    ends_excl = np.r_[changes[1:], seq.size]
    labels = seq[starts]

    rows = []
    for s, e_excl, lab in zip(starts, ends_excl, labels):
        if lab == 0:  # skip transitions
            continue
        e_incl = e_excl - 1
        rows.append({
            "fix_start": int(s),
            "fix_end": int(e_incl),
            "fix_location": int(lab)
        })

    return pd.DataFrame(rows)


def derasterize_fixations(
    df_long: pd.DataFrame,
    start_col: str = "fix_start",   # inclusive
    end_col: str = "fix_end",       # inclusive
    loc_col: str = "fix_location",
    fill_code: int = 0,
    dtype=np.int8,
) -> np.ndarray:
    """
    Single-trial: long DF of fixations -> sequence (1D array of ints).

    Parameters
    ----------
    df_long : pd.DataFrame
        Must have columns [loc_col, start_col, end_col].
        One row per fixation segment.
    start_col, end_col, loc_col : str
        Column names for start (inclusive), end (inclusive), and location code.
    fill_code : int
        Code used where there is no fixation (gaps).
    dtype : np.dtype
        dtype of the returned 1D array.

    Returns
    -------
    np.ndarray
        1D array for the single trial (length = max(fix_end) + 1),
        or empty array if df_long is empty.
    """
    if df_long.empty:
        return np.array([], dtype=dtype)

    g = df_long[[start_col, end_col, loc_col]].copy()
    g[start_col] = g[start_col].astype(int)
    g[end_col]   = g[end_col].astype(int)
    g[loc_col]   = g[loc_col].astype(int)
    g = g.sort_values(start_col)

    L = int(g[end_col].max()) + 1
    if L <= 0:
        return np.array([], dtype=dtype)

    seq = np.full(L, fill_code, dtype=dtype)

    for _, r in g.iterrows():
        s, e, lab = int(r[start_col]), int(r[end_col]), int(r[loc_col])
        if s > e or s >= L:
            continue
        s2 = max(s, 0)
        e2 = min(e, L - 1)              # inclusive clamp
        seq[s2:e2 + 1] = lab            # inclusive slice

    return seq
