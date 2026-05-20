import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Config (edit as needed) ----------
# Directory with CSV files (falls back to /mnt/data for notebook uploads)
DATA_DIR = Path.cwd()
if not DATA_DIR.exists():
    DATA_DIR = Path("/mnt/data")

# File pattern to process
FILE_GLOB = "*.csv"

# Output
OUTPUT_CSV = DATA_DIR / "ccf_window_summary.csv"

# Timezone handling for timestamps
TZ_USED = "UTC"          # e.g., "UTC" or "America/New_York"

# Cross-correlation lag window (minutes, ±)
MAX_LAG_MIN = 30

# Timestamp column
TIMESTAMP_COL = "t_stamp"

# Analysis windows:
# - If WINDOW_COLUMN is set, each unique value in that column defines one window.
# - Otherwise, WINDOW_DEFINITIONS defines timestamp-based windows.
# - start/end may be None, absolute timestamp strings, or pandas Timedelta strings
#   relative to the first timestamp in each file.
WINDOW_COLUMN = None
WINDOW_DEFINITIONS = [
    {"label": "window_1", "start": None, "end": None},
]

# Column identification:
# If you know your timestamp and two value columns, set regex patterns here.
# If patterns are None, the script will auto-detect:
#  - timestamp column named by TIMESTAMP_COL
#  - first two numeric columns (excluding TIMESTAMP_COL) as y_col and x_col
Y_COL_REGEX = None  # e.g., r"(?i)^measurement_a$"
X_COL_REGEX = None  # e.g., r"(?i)^measurement_b$"

# Optional upper bound filter for Y values; set to None to disable
Y_UPPER_BOUND = None  # e.g., set a numeric maximum or leave as None

# Optional filename metadata extraction:
# If your filenames encode two identifiers, set a regex with two capture groups.
# If None or not matched, identifiers are left empty.
FILENAME_ID_REGEX = None  # e.g., r"^(.+)_(.+)$"


# ---------- Helpers ----------
def find_unique_col_regex(cols, pattern, label):
    cand = [c for c in cols if re.search(pattern, c, flags=re.I)]
    if len(cand) != 1:
        raise ValueError(f"Could not uniquely identify {label}. Found: {cand}")
    return cand[0]

def autodetect_value_columns(df, ts_col=TIMESTAMP_COL, exclude_cols=None):
    """Pick the first two numeric columns (excluding timestamp)."""
    excluded = {ts_col}
    if exclude_cols:
        excluded.update(exclude_cols)

    num_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        # be tolerant: coerce-to-numeric and retry
        coerce_ok = []
        for c in df.columns:
            if c in excluded:
                continue
            try:
                pd.to_numeric(df[c], errors="coerce")
                coerce_ok.append(c)
            except Exception:
                pass
        num_cols = coerce_ok
    if len(num_cols) < 2:
        raise ValueError("Could not autodetect two numeric value columns.")
    return num_cols[0], num_cols[1]

def zscore_arr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    return (x - mu) / sd if sd > 0 else np.full_like(x, np.nan)

def ccf_pearson(x: np.ndarray, y: np.ndarray, lag_max: int) -> dict[int, float]:
    """
    R-like ccf(x, y): correlation between x_{t+k} and y_t.
    Interpretation: k>0 means x follows y by k samples.
    """
    lags = range(-lag_max, lag_max + 1)
    out = {}
    n = len(x)
    for k in lags:
        if k >= 0:
            xs, ys = x[k:], y[:n-k]
        else:
            xs, ys = x[:n+k], y[-k:]
        out[k] = np.corrcoef(xs, ys)[0, 1] if len(xs) >= 2 else np.nan
    return out

def parse_two_ids_from_filename(path: Path) -> tuple[object, object]:
    """
    Try to extract two identifiers from the filename using FILENAME_ID_REGEX.
    Returns (id1, id2) or (None, None) if not matched or not configured.
    """
    if not FILENAME_ID_REGEX:
        return None, None
    stem = path.stem
    m = re.match(FILENAME_ID_REGEX, stem) or re.search(FILENAME_ID_REGEX, stem)
    if not m or len(m.groups()) < 2:
        return None, None
    return m.group(1), m.group(2)

def _resolve_window_bound(value, t0, tz_used: str):
    """Resolve a configured window boundary to a timestamp."""
    if value is None:
        return None
    try:
        return t0 + pd.to_timedelta(value)
    except (TypeError, ValueError):
        pass

    ts = pd.to_datetime(value, errors="raise")
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_used)
    else:
        ts = ts.tz_convert(tz_used)
    return ts

def build_window_masks(df, tz_used: str):
    """Build named boolean masks from a data column or configured timestamp bounds."""
    if WINDOW_COLUMN:
        if WINDOW_COLUMN not in df.columns:
            raise ValueError(f"Window column '{WINDOW_COLUMN}' not found.")
        values = df[WINDOW_COLUMN].dropna().unique()
        windows = []
        for value in values:
            label = str(value)
            windows.append((label, df[WINDOW_COLUMN].eq(value)))
        if not windows:
            raise ValueError(f"Window column '{WINDOW_COLUMN}' contains no usable values.")
        return windows

    if not WINDOW_DEFINITIONS:
        raise ValueError("WINDOW_DEFINITIONS must contain at least one window.")

    t0 = df[TIMESTAMP_COL].iloc[0]
    windows = []
    for idx, spec in enumerate(WINDOW_DEFINITIONS, start=1):
        label = str(spec.get("label") or f"window_{idx}")
        start = _resolve_window_bound(spec.get("start"), t0, tz_used)
        end = _resolve_window_bound(spec.get("end"), t0, tz_used)
        if start is not None and end is not None and start > end:
            raise ValueError(f"Window '{label}' has a start after its end.")

        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df[TIMESTAMP_COL] >= start
        if end is not None:
            mask &= df[TIMESTAMP_COL] <= end
        windows.append((label, mask))

    return windows

def _segment_analysis(df, y_col, x_col, base_mask, seg_name, seg_mask, max_lag_minutes):
    mask_seg = base_mask & seg_mask
    n_total_seg = int(seg_mask.sum())
    n_kept = int(mask_seg.sum())

    # Optional count of excluded Y values above threshold (if configured)
    excluded_y_gt_thresh = 0
    if Y_UPPER_BOUND is not None:
        y_vals_seg = pd.to_numeric(df.loc[seg_mask, y_col], errors="coerce")
        excluded_y_gt_thresh = int(np.nansum(y_vals_seg > Y_UPPER_BOUND))

    if n_kept < 10:
        return {
            "status": "too_few_samples",
            "segment": seg_name,
            "n_total_seg": n_total_seg,
            "n_kept": n_kept,
            "excluded_y_gt_thresh": excluded_y_gt_thresh,
        }

    y_kept = pd.to_numeric(df.loc[mask_seg, y_col], errors="coerce").to_numpy()
    x_kept = pd.to_numeric(df.loc[mask_seg, x_col], errors="coerce").to_numpy()
    t_kept = df.loc[mask_seg, TIMESTAMP_COL].to_numpy()

    y_z = zscore_arr(y_kept)
    x_z = zscore_arr(x_kept)

    if len(t_kept) >= 2:
        diffs_min = np.diff(t_kept).astype("timedelta64[s]").astype(float) / 60.0
        dt_min = float(np.nanmedian(diffs_min)) if np.isfinite(diffs_min).any() else 1.0
    else:
        dt_min = 1.0
    if not np.isfinite(dt_min) or dt_min <= 0:
        dt_min = 1.0
    lag_max_samples = max(1, int(round(max_lag_minutes / dt_min)))

    corrs = ccf_pearson(x_z, y_z, lag_max=lag_max_samples)
    valid = {k: v for k, v in corrs.items() if np.isfinite(v)}
    if not valid:
        return {
            "status": "no_valid_correlations",
            "segment": seg_name,
            "n_total_seg": n_total_seg,
            "n_kept": n_kept,
            "excluded_y_gt_thresh": excluded_y_gt_thresh,
        }

    best_lag = max(valid, key=lambda k: abs(valid[k]))
    best_corr = valid[best_lag]
    best_minutes = best_lag * dt_min

    if best_lag > 0:
        direction = "x_follows_y"
    elif best_lag < 0:
        direction = "y_follows_x"
    else:
        direction = "synchronous"

    return {
        "status": "ok",
        "segment": seg_name,
        "time_start": pd.to_datetime(t_kept[0]).isoformat(),
        "time_end": pd.to_datetime(t_kept[-1]).isoformat(),
        "cadence_min": dt_min,
        "lag_window_samples": lag_max_samples,
        "max_abs_corr": float(abs(best_corr)),
        "signed_corr": float(best_corr),
        "best_lag_samples": int(best_lag),
        "best_lag_minutes": float(best_minutes),
        "direction": direction,
        "n_total_seg": n_total_seg,
        "n_kept": n_kept,
        "excluded_y_gt_thresh": excluded_y_gt_thresh,
    }

def _delta_row(common_meta, previous_result, current_result):
    """
    Compare the current window minus the previous window for absolute corr and lag metrics.
    If either segment is not 'ok', emit a row with status explaining why.
    """
    row = dict(common_meta)
    previous_label = None if not previous_result else previous_result.get("segment")
    current_label = None if not current_result else current_result.get("segment")
    row["segment"] = f"delta:{current_label}-minus-{previous_label}"

    if (
        not previous_result
        or not current_result
        or previous_result.get("status") != "ok"
        or current_result.get("status") != "ok"
    ):
        previous_status = None if not previous_result else previous_result.get("status")
        current_status = None if not current_result else current_result.get("status")
        row.update({
            "status": f"insufficient_segments: previous={previous_status}, current={current_status}",
            "delta_abs_corr": np.nan,
            "delta_lag_minutes": np.nan,
            "delta_lag_samples": np.nan,
        })
        return row

    delta_abs_corr = float(current_result["max_abs_corr"] - previous_result["max_abs_corr"])
    delta_lag_minutes = float(current_result["best_lag_minutes"] - previous_result["best_lag_minutes"])
    delta_lag_samples = int(current_result["best_lag_samples"] - previous_result["best_lag_samples"])

    row.update({
        "status": "ok",
        "delta_abs_corr": delta_abs_corr,
        "delta_lag_minutes": delta_lag_minutes,
        "delta_lag_samples": delta_lag_samples,
        # Per-segment fields not relevant for delta row
        "time_start": None, "time_end": None,
        "cadence_min": None, "lag_window_samples": None,
        "max_abs_corr": None, "signed_corr": None,
        "best_lag_samples": None, "best_lag_minutes": None,
        "direction": None,
        "n_total_seg": None, "n_kept": None, "excluded_y_gt_thresh": None,
    })
    return row

def ccf_on_file_windows(ts_csv: Path, id1=None, id2=None, tz_used: str = "UTC", max_lag_minutes: int = 30):
    try:
        df = pd.read_csv(ts_csv)
        if TIMESTAMP_COL not in df.columns:
            raise ValueError(f"Column '{TIMESTAMP_COL}' not found.")

        t = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=False)
        if t.dt.tz is None:
            t = t.dt.tz_localize(tz_used)
        else:
            t = t.dt.tz_convert(tz_used)

        df = df.assign(**{TIMESTAMP_COL: t}).sort_values(TIMESTAMP_COL).dropna(subset=[TIMESTAMP_COL])
        if df.empty:
            raise ValueError("No valid timestamps after parsing.")

        # Identify Y and X columns
        if Y_COL_REGEX:
            y_col = find_unique_col_regex(df.columns, Y_COL_REGEX, "Y column")
        else:
            y_col = None
        if X_COL_REGEX:
            x_col = find_unique_col_regex(df.columns, X_COL_REGEX, "X column")
        else:
            x_col = None

        if not (y_col and x_col):
            # Autodetect if patterns not set or not found
            exclude_cols = [WINDOW_COLUMN] if WINDOW_COLUMN else None
            y_col_auto, x_col_auto = autodetect_value_columns(
                df,
                ts_col=TIMESTAMP_COL,
                exclude_cols=exclude_cols,
            )
            y_col = y_col or y_col_auto
            x_col = x_col or x_col_auto

        y_raw = pd.to_numeric(df[y_col], errors="coerce")
        x_raw = pd.to_numeric(df[x_col], errors="coerce")

        base_mask = y_raw.notna() & x_raw.notna()
        if Y_UPPER_BOUND is not None:
            base_mask &= (y_raw <= Y_UPPER_BOUND)

        windows = build_window_masks(df, tz_used)
        segment_results = [
            _segment_analysis(df, y_col, x_col, base_mask, label, mask, max_lag_minutes)
            for label, mask in windows
        ]

        common = {
            "file": ts_csv.name,
            "ID1": id1,
            "ID2": id2,
            "x_column": x_col,
            "y_column": y_col,
            "time_full_start": df[TIMESTAMP_COL].iloc[0].isoformat(),
            "time_full_end": df[TIMESTAMP_COL].iloc[-1].isoformat(),
        }

        rows = []
        for seg_res in segment_results:
            row = dict(common)
            row.update(seg_res)
            rows.append(row)

        for previous_result, current_result in zip(segment_results, segment_results[1:]):
            rows.append(_delta_row(common, previous_result, current_result))
        return rows

    except Exception as e:
        base_error = {
            "file": ts_csv.name,
            "ID1": id1,
            "ID2": id2,
            "status": f"error: {e}",
            "x_column": None,
            "y_column": None,
            "time_full_start": None,
            "time_full_end": None,
        }
        return [dict(base_error, segment=None)]

def main():
    all_rows = []
    csv_files = sorted(DATA_DIR.glob(FILE_GLOB))
    print(f"Found {len(csv_files)} '{FILE_GLOB}' files in {DATA_DIR}")

    for f in csv_files:
        id1, id2 = parse_two_ids_from_filename(f)
        if id1 is not None or id2 is not None:
            print(f"Processing {f.name}  →  IDs: {id1}, {id2}")
        else:
            print(f"Processing {f.name}")

        rows = ccf_on_file_windows(f, id1=id1, id2=id2, tz_used=TZ_USED, max_lag_minutes=MAX_LAG_MIN)
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)

    col_order = [
        "file", "ID1", "ID2", "segment", "status",
        "x_column", "y_column",
        "time_full_start", "time_full_end",
        "time_start", "time_end",
        "cadence_min", "lag_window_samples",
        "max_abs_corr", "signed_corr",
        "best_lag_samples", "best_lag_minutes",
        "direction",
        "n_total_seg", "n_kept", "excluded_y_gt_thresh",
        "delta_abs_corr", "delta_lag_minutes", "delta_lag_samples",
    ]
    summary_df = summary_df.reindex(columns=col_order + [c for c in summary_df.columns if c not in col_order])

    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
