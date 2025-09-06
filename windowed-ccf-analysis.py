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
OUTPUT_CSV = DATA_DIR / "ccf_two_window_summary.csv"

# Timezone handling for timestamps
TZ_USED = "UTC"          # e.g., "UTC" or "America/New_York"

# Cross-correlation lag window (minutes, ±)
MAX_LAG_MIN = 30

# Two analysis windows relative to the first timestamp
# Window 1: [t0+k, t0+k]; Window 2: [t0+k, end]
SEG1_LABEL = "SEG1"
SEG2_LABEL = "SEG2"
SEG3_LABEL = "delta"

# Column identification:
# If you know your timestamp and two value columns, set regex patterns here.
# If patterns are None, the script will auto-detect:
#  - timestamp column named 't_stamp' (required)
#  - first two numeric columns (excluding 't_stamp') as y_col and x_col (in that order)
Y_COL_REGEX = None  # e.g., r"(?i)humidity|mist"
X_COL_REGEX = None  # e.g., r"(?i)temperature|fog"

# Optional upper bound filter for Y values; set to None to disable
Y_UPPER_BOUND = None  # e.g., 100 for domain-specific filtering, else None

# Optional filename metadata extraction:
# If your filenames encode two identifiers, set a regex with two capture groups.
# If None or not matched, identifiers are left empty.
FILENAME_ID_REGEX = None  # e.g., r'(?i)^BH(\d+)([A-Z])$'


# ---------- Helpers ----------
def find_unique_col_regex(cols, pattern, label):
    cand = [c for c in cols if re.search(pattern, c, flags=re.I)]
    if len(cand) != 1:
        raise ValueError(f"Could not uniquely identify {label}. Found: {cand}")
    return cand[0]

def autodetect_value_columns(df, ts_col="t_stamp"):
    """Pick the first two numeric columns (excluding timestamp)."""
    num_cols = [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        # be tolerant: coerce-to-numeric and retry
        coerce_ok = []
        for c in df.columns:
            if c == ts_col:
                continue
            try:
                pd.to_numeric(df[c], errors="coerce")
                coerce_ok.append(c)
            except Exception:
                pass
        num_cols = coerce_ok
    if len(num_cols) < 2:
        raise ValueError("Could not autodetect two numeric value columns.")
    # Treat the first as Y and the second as X (consistent and documented)
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
    t_kept = df.loc[mask_seg, "t_stamp"].to_numpy()

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

def _delta_row(common_meta, res1, res2):
    """
    Compare SEG2 minus SEG1 for absolute corr and lag metrics.
    If either segment is not 'ok', emit a row with status explaining why.
    """
    row = dict(common_meta)
    row["segment"] = SEG3_LABEL

    if not res1 or not res2 or res1.get("status") != "ok" or res2.get("status") != "ok":
        r1s = None if not res1 else res1.get("status")
        r2s = None if not res2 else res2.get("status")
        row.update({
            "status": f"insufficient_segments: seg1={r1s}, seg2={r2s}",
            "delta_abs_corr": np.nan,
            "delta_lag_minutes": np.nan,
            "delta_lag_samples": np.nan,
        })
        return row

    delta_abs_corr = float(res2["max_abs_corr"] - res1["max_abs_corr"])
    delta_lag_minutes = float(res2["best_lag_minutes"] - res1["best_lag_minutes"])
    delta_lag_samples = int(res2["best_lag_samples"] - res1["best_lag_samples"])

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

def ccf_on_file_two_windows(ts_csv: Path, id1=None, id2=None, tz_used: str = "UTC", max_lag_minutes: int = 30):
    try:
        df = pd.read_csv(ts_csv)
        if "t_stamp" not in df.columns:
            raise ValueError("Column 't_stamp' not found.")

        t = pd.to_datetime(df["t_stamp"], errors="coerce", utc=False)
        if t.dt.tz is None:
            t = t.dt.tz_localize(tz_used)
        else:
            t = t.dt.tz_convert(tz_used)

        df = df.assign(t_stamp=t).sort_values("t_stamp").dropna(subset=["t_stamp"])
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
            y_col_auto, x_col_auto = autodetect_value_columns(df, ts_col="t_stamp")
            y_col = y_col or y_col_auto
            x_col = x_col or x_col_auto

        y_raw = pd.to_numeric(df[y_col], errors="coerce")
        x_raw = pd.to_numeric(df[x_col], errors="coerce")

        base_mask = y_raw.notna() & x_raw.notna()
        if Y_UPPER_BOUND is not None:
            base_mask &= (y_raw <= Y_UPPER_BOUND)

        t0 = df["t_stamp"].iloc[0]
        start_10h = t0 + pd.Timedelta(hours=10)
        end_72h   = t0 + pd.Timedelta(hours=72)

        seg1_mask = (df["t_stamp"] >= start_10h) & (df["t_stamp"] <= end_72h)   # 10–72 h
        seg2_mask = (df["t_stamp"] >= end_72h)                                   # 72 h → end

        res1 = _segment_analysis(df, y_col, x_col, base_mask, SEG1_LABEL, seg1_mask, max_lag_minutes)
        res2 = _segment_analysis(df, y_col, x_col, base_mask, SEG2_LABEL, seg2_mask, max_lag_minutes)

        common = {
            "file": ts_csv.name,
            "ID1": id1,
            "ID2": id2,
            "x_column": x_col,
            "y_column": y_col,
            "time_full_start": df["t_stamp"].iloc[0].isoformat(),
            "time_full_end": df["t_stamp"].iloc[-1].isoformat(),
        }

        rows = []
        for seg_res in (res1, res2):
            row = dict(common)
            row.update(seg_res)
            rows.append(row)

        rows.append(_delta_row(common, res1, res2))
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
        return [
            dict(base_error, segment=SEG1_LABEL),
            dict(base_error, segment=SEG2_LABEL),
            dict(base_error, segment=SEG3_LABEL, delta_abs_corr=np.nan, delta_lag_minutes=np.nan, delta_lag_samples=np.nan),
        ]

# ---------- Batch over directory & export ----------
all_rows = []
csv_files = sorted(DATA_DIR.glob(FILE_GLOB))
print(f"Found {len(csv_files)} '{FILE_GLOB}' files in {DATA_DIR}")

for f in csv_files:
    id1, id2 = parse_two_ids_from_filename(f)
    if id1 is not None or id2 is not None:
        print(f"Processing {f.name}  →  IDs: {id1}, {id2}")
    else:
        print(f"Processing {f.name}")

    rows = ccf_on_file_two_windows(f, id1=id1, id2=id2, tz_used=TZ_USED, max_lag_minutes=MAX_LAG_MIN)
    all_rows.extend(rows)

summary_df = pd.DataFrame(all_rows)

# Order columns (include delta columns at end)
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
