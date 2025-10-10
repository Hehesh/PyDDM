import numpy as np
import pandas as pd

# List to DF
def rasterize_fixations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each trial's fixation sequence (a list/array of codes) into
    (trial, fix_num, location, fix_start, fix_end, fix_dur) rows.

    Expects:
        df['fixation'] : list/np.ndarray per trial (e.g., [0,1,1,2,...])

    Returns:
        long-format DataFrame with one row per fixation.
    """
    rows = []
    df = df.copy()
    if 'trial' not in df.columns:
        df['trial'] = np.arange(1, len(df) + 1)

    for _, row in df.iterrows():
        seq = np.asarray(row['fixation'], dtype=np.int64)
        if seq.size == 0:
            continue

        # run-length encode
        changes = np.flatnonzero(np.diff(seq, prepend=seq[0] - 1))
        starts = changes
        ends = np.r_[changes[1:], seq.size]
        labels = seq[starts]

        mask = (labels == 1) | (labels == 2)
        starts, ends, labels = starts[mask], ends[mask], labels[mask]
        if starts.size == 0:
            continue

        for fix_num, (s, e, lab) in enumerate(zip(starts, ends, labels), start=1):
            rows.append({
                'trial': row['trial'],
                'choice': row.get('choice', np.nan),
                'RT': row.get('RT', np.nan),
                'avgWTP_left': row.get('avgWTP_left', np.nan),
                'avgWTP_right': row.get('avgWTP_right', np.nan),
                'fix_num': fix_num,
                'location': int(lab),
                'fix_start': int(s),
                'fix_end': int(e),
                'fix_dur': int(e - s)
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out['fix_num_rev'] = out.groupby('trial')['fix_num'].transform(
            lambda x: x.max() - x + 1
        )
    return out


# DF to list
def derasterize_fixations(
    df_long: pd.DataFrame,
    *,
    trial_col: str = "trial",
    start_col: str = "fix_start",   # inclusive
    end_col: str = "fix_end",       # exclusive
    loc_col: str = "location",
    fill_code: int = 0,
    dtype=np.int8,
    length_map: dict | None = None, # optional {trial_id: total_length}
    pad: bool = False,              # if True, returns a 2D padded array
    pad_value: int = 0
) -> np.ndarray:
    """
    Inverse of rasterize_fixations: long DF -> per-trial binned sequences.

    Expects df_long to contain rows for a single dataset with columns:
      trial_col, loc_col, start_col, end_col  (end exclusive).
    Returns:
      - if pad=False: np.ndarray(dtype=object) where each element is 1D np.ndarray[int]
      - if pad=True:  np.ndarray shape (n_trials, max_len) padded with pad_value
    """
    if df_long.empty:
        return np.array([], dtype=object) if not pad else np.zeros((0, 0), dtype=dtype)

    work = df_long[[trial_col, start_col, end_col, loc_col]].copy()
    work[start_col] = work[start_col].astype(int)
    work[end_col]   = work[end_col].astype(int)
    work[loc_col]   = work[loc_col].astype(int)

    seqs = []
    lengths = []
    trial_ids = []

    for tid, g in work.sort_values([trial_col, start_col]).groupby(trial_col, sort=False):
        # Determine sequence length
        if length_map is not None and tid in length_map:
            L = int(length_map[tid])
        else:
            L = int(g[end_col].max()) if not g.empty else 0

        seq = np.full(L, fill_code, dtype=dtype)

        # Paint fixations; later rows overwrite earlier on overlaps
        for _, r in g.iterrows():
            s = int(r[start_col]); e = int(r[end_col]); lab = int(r[loc_col])
            if L == 0 or s >= L or e <= 0 or s >= e:
                continue
            s2, e2 = max(s, 0), min(e, L)
            if s2 < e2:
                seq[s2:e2] = lab

        seqs.append(seq)
        lengths.append(L)
        trial_ids.append(tid)

    if not pad:
        # Variable-length sequences as object array
        return np.array(seqs, dtype=object)

    # Pad to 2D array
    max_len = max(lengths) if lengths else 0
    out = np.full((len(seqs), max_len), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, :s.size] = s
    return out