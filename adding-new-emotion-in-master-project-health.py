#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

EMOS = ["frustration", "satisfaction", "neutral", "caution"]

# ---------- dates → period ----------
def _coerce_dates_any(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce", utc=True)

    mask = dt.isna()
    if mask.any():
        # e.g., "Wed Sep 25 20:35:23 2024 +0000"
        dt_git = pd.to_datetime(
            s[mask], format="%a %b %d %H:%M:%S %Y %z", errors="coerce", utc=True
        )
        dt.loc[mask] = dt_git
        mask = dt.isna()

    if mask.any():
        m10 = s[mask].str.fullmatch(r"\d{10}")
        if m10.any():
            idx = m10[m10].index
            dt.loc[idx] = pd.to_datetime(s.loc[idx], unit="s", utc=True)
            mask = dt.isna()

    if mask.any():
        m13 = s[mask].str.fullmatch(r"\d{13}")
        if m13.any():
            idx = m13[m13].index
            dt.loc[idx] = pd.to_datetime(s.loc[idx], unit="ms", utc=True)

    return dt

def add_time_period(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    dt = _coerce_dates_any(df[date_col])
    out = df.copy()
    out["_dt"] = dt.dt.tz_convert(None)
    f = freq.upper()

    if f == "Q":
        s = out["_dt"].dt.to_period("Q").astype("string")      # 'YYYYQn'
        out["time_period"] = s.str.replace(r"(\d{4})Q(\d)", r"\1-Q\2", regex=True)
    elif f == "M":
        out["time_period"] = out["_dt"].dt.to_period("M").astype("string")  # 'YYYY-MM'
    elif f == "W":
        iso = out["_dt"].dt.isocalendar()
        yr = iso.year.astype("Int64").astype("string")
        wk = iso.week.astype("Int64").astype("string").str.zfill(2)
        out["time_period"] = (yr + "-W" + wk).astype("string")
    else:
        raise ValueError("freq must be one of Q/M/W")

    return out.drop(columns=["_dt"]).loc[out["time_period"].notna()].copy()

# ---------- emotion metrics ----------
def _norm_emotion(x):
    if pd.isna(x):
        return x
    return str(x).strip().lower()

def _safe_ratio(n, d):
    try:
        return float(n) / float(d) if d else np.nan
    except Exception:
        return np.nan

def compute_emotion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["emotion_predicted"] = d["emotion_predicted"].map(_norm_emotion)

    # masks + per-emotion score means
    for e in EMOS:
        d[f"is_{e}"] = (d["emotion_predicted"] == e).astype("Int64")
        d[f"{e}_score_mask"] = np.where(
            d["emotion_predicted"] == e, d.get("emotion_score"), np.nan
        )

    total_col = "commit" if "commit" in d.columns else "emotion_predicted"
    g = d.groupby(["repo", "time_period"], dropna=False).agg(
        total_commits_emotions=(total_col, "count"),
        frustration_count=("is_frustration", "sum"),
        satisfaction_count=("is_satisfaction", "sum"),
        neutral_count=("is_neutral", "sum"),
        caution_count=("is_caution", "sum"),
        avg_frustration_score=("frustration_score_mask", "mean"),
        avg_satisfaction_score=("satisfaction_score_mask", "mean"),
        avg_neutral_score=("neutral_score_mask", "mean"),
        avg_caution_score=("caution_score_mask", "mean"),
        avg_emotion_score_overall=("emotion_score", "mean"),
    ).reset_index()

    # ratios
    g["frustration_ratio"] = g.apply(
        lambda r: _safe_ratio(r["frustration_count"], r["total_commits_emotions"]),
        axis=1,
    )
    g["satisfaction_ratio"] = g.apply(
        lambda r: _safe_ratio(r["satisfaction_count"], r["total_commits_emotions"]),
        axis=1,
    )
    g["neutral_ratio"] = g.apply(
        lambda r: _safe_ratio(r["neutral_count"], r["total_commits_emotions"]), axis=1
    )
    g["caution_ratio"] = g.apply(
        lambda r: _safe_ratio(r["caution_count"], r["total_commits_emotions"]), axis=1
    )

    cols = [
        "repo",
        "time_period",
        "total_commits_emotions",
        "frustration_count",
        "frustration_ratio",
        "avg_frustration_score",
        "satisfaction_count",
        "satisfaction_ratio",
        "avg_satisfaction_score",
        "neutral_count",
        "neutral_ratio",
        "avg_neutral_score",
        "caution_count",
        "caution_ratio",
        "avg_caution_score",
        "avg_emotion_score_overall",
    ]
    return g[cols].sort_values(["repo", "time_period"]).reset_index(drop=True)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Repo × period emotion metrics. Defaults to commits_with_emotions_final.csv → emotion_repo_quarter.csv (quarterly)."
    )
    parser.add_argument(
        "--in",
        dest="inp",
        default="commits_with_emotions_final.csv",
        help="Input CSV (default: commits_with_emotions_final.csv)",
    )
    parser.add_argument(
        "--freq",
        default="Q",
        choices=["Q", "M", "W"],
        help="Period: Q (default), M, W",
    )
    parser.add_argument(
        "--out",
        default="emotion_repo_quarter.csv",
        help="Output CSV (default: emotion_repo_quarter.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    need = {"repo", "date", "emotion_predicted", "emotion_score"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(
            f"Missing required columns in {args.inp}: {sorted(miss)}"
        )

    df = add_time_period(df, "date", args.freq)
    out = compute_emotion_metrics(df)
    out.to_csv(args.out, index=False)

    # quick terminal summary
    print(f"Wrote: {args.out} (freq={args.freq})")
    print(f"Repos: {out['repo'].nunique()}, Rows: {len(out)}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
