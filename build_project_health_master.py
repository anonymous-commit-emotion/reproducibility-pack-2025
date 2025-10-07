#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re
import numpy as np
import pandas as pd

# ----------------------------
# Config / constants
# ----------------------------
BUG_KEYWORDS = [
    "fix","fixes","fixed","fixing",
    "bug","bugs",
    "issue","issues",
    "error","errors",
    "patch","patched","patches",
    "resolve","resolves","resolved","resolving",
    "correct","corrected","correcting","correction",
]
EMOS = ["frustration","satisfaction","neutral","caution"]
CHURN_COLS = ["avg_files_changed","avg_lines_added","avg_lines_deleted","commits_with_churn"]

# pBIC heuristics
FAILURE_KEYWORDS = [
    "oops","broke","broken","breakage",
    "regression","regressed",
    "fail","failed","failure",
    "crash","crashed","crashing",
    "introduce bug","introduced bug","caused bug"
]
NEGATIVE_EMOS = {"frustration","caution"}

bug_pattern       = re.compile(r"\b(" + "|".join(BUG_KEYWORDS) + r")\b", re.IGNORECASE)
revert_pattern    = re.compile(r"\b(revert(?:ed|ing)?|revert of|rollback|back[- ]?out)\b", re.IGNORECASE)
failure_kw_pattern= re.compile(r"\b(" + "|".join(map(re.escape, FAILURE_KEYWORDS)) + r")\b", re.IGNORECASE)

# ----------------------------
# Key normalization helpers
# ----------------------------
def _coerce_dates_any(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    mask = dt.isna()
    if mask.any():
        dt_git = pd.to_datetime(s[mask], format="%a %b %d %H:%M:%S %Y %z", errors="coerce", utc=True)
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
    if f == "M":
        out["time_period"] = out["_dt"].dt.to_period("M").astype("string")
    elif f == "W":
        cal = out["_dt"].dt.isocalendar()
        out["time_period"] = (cal.year.astype("Int64").astype("string")
                              + "-W" + cal.week.astype("Int64").astype("string").str.zfill(2))
    elif f == "Q":
        s = out["_dt"].dt.to_period("Q").astype("string")  # 'YYYYQn'
        out["time_period"] = s.str.replace(r"(\d{4})Q(\d)", r"\1-Q\2", regex=True)
    else:
        raise ValueError("freq must be Q/M/W")
    return out.drop(columns=["_dt"]).loc[out["time_period"].notna()].copy()

def _repo_norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.casefold()

def _tp_to_reference_date(tp: str):
    if tp is None or (isinstance(tp, float) and np.isnan(tp)): return None
    s = str(tp).strip()

    m = re.fullmatch(r"(?P<y>\d{4})-?Q(?P<q>[1-4])", s, re.IGNORECASE)
    if m:
        y, q = int(m.group("y")), int(m.group("q"))
        return pd.Timestamp(year=y, month=1 + (q-1)*3, day=1)

    m = re.fullmatch(r"(?P<y>\d{4})-(?P<m>\d{1,2})", s)
    if m:
        y, mo = int(m.group("y")), int(m.group("m"))
        mo = max(1, min(12, mo))
        return pd.Timestamp(year=y, month=mo, day=1)

    m = re.fullmatch(r"(?P<y>\d{4})-?W(?P<w>\d{2})", s, re.IGNORECASE)
    if m:
        y, w = int(m.group("y")), int(m.group("w"))
        try:
            return pd.to_datetime(f"{y}-W{w}-1", format="%G-W%V-%u")
        except Exception:
            return None

    try:
        return pd.to_datetime(s, errors="raise").tz_localize(None)
    except Exception:
        return None

def _normalize_tp_series_to_freq(tp_series: pd.Series, target_freq: str) -> pd.Series:
    target = target_freq.upper()
    ref_dates = tp_series.apply(_tp_to_reference_date)
    if target == "Q":
        s = ref_dates.dt.to_period("Q").astype("string")
        return s.str.replace(r"(\d{4})Q(\d)", r"\1-Q\2", regex=True)
    if target == "M":
        return ref_dates.dt.to_period("M").astype("string")
    if target == "W":
        cal = ref_dates.dt.isocalendar()
        return (cal.year.astype("Int64").astype("string")
                + "-W" + cal.week.astype("Int64").astype("string").str.zfill(2)).astype("string")
    raise ValueError("target_freq must be Q/M/W")

# ----------------------------
# Tasks 1–3 (with pBICs)
# ----------------------------
def compute_task1_metrics(df: pd.DataFrame, pbic_score_threshold: float = 0.70) -> pd.DataFrame:
    d = df.copy()
    d["message"] = d["message"].astype(str)

    # pBFC & revert via keywords
    d["is_pBFC"]    = d["message"].apply(lambda m: bool(bug_pattern.search(m)))
    d["is_pRevert"] = d["message"].apply(lambda m: bool(revert_pattern.search(m)))

    # failure keywords
    d["has_failure_kw"] = d["message"].apply(lambda m: bool(failure_kw_pattern.search(m)))

    # negative emotion (frustration/caution or high score)
    emo   = d.get("emotion_predicted")
    score = d.get("emotion_score")
    if emo is None or score is None:
        d["is_negative_emotion"] = False
    else:
        emo = emo.astype(str).str.strip().str.lower()
        scr = pd.to_numeric(score, errors="coerce")
        d["is_negative_emotion"] = emo.isin(NEGATIVE_EMOS) | (scr >= pbic_score_threshold)

    # pBIC: negative signal AND NOT pBFC
    d["is_pBIC"] = (d["is_negative_emotion"] | d["has_failure_kw"]) & (~d["is_pBFC"])

    grp = d.groupby(["repo","time_period"], dropna=False).agg(
        pBFC_rate=("is_pBFC","mean"),
        pRevert_rate=("is_pRevert","mean"),
        pBIC_rate=("is_pBIC","mean"),
        pBIC_count=("is_pBIC","sum"),
    ).reset_index()
    return grp

def compute_task2_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    author_col = "author" if "author" in d.columns else ("author_email" if "author_email" in d.columns else None)
    if author_col is None:
        author_col = "__author__"; d[author_col] = None
    commit_col = "commit" if "commit" in d.columns else "message"
    g = d.groupby(["repo","time_period"], dropna=False).agg(
        total_commits_prod=(commit_col,"count"),
        unique_authors=(author_col,"nunique"),
    ).reset_index()
    g["avg_commits_per_author"] = g.apply(lambda r: (r["total_commits_prod"]/r["unique_authors"]) if r["unique_authors"] else 0.0, axis=1)
    return g

def _norm_emotion(x):
    if pd.isna(x): return x
    return str(x).strip().lower()

def _safe_ratio(n,d):
    try: return float(n)/float(d) if d else np.nan
    except: return np.nan

def compute_task3_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["emotion_predicted"] = d["emotion_predicted"].map(_norm_emotion)

    for e in EMOS:
        d[f"is_{e}"] = (d["emotion_predicted"] == e).astype("Int64")
        d[f"{e}_score_mask"] = np.where(d["emotion_predicted"] == e, d.get("emotion_score"), np.nan)

    total_col = "commit" if "commit" in d.columns else "emotion_predicted"
    g = d.groupby(["repo","time_period"], dropna=False).agg(
        total_commits_emotions=(total_col,"count"),
        frustration_count=("is_frustration","sum"),
        satisfaction_count=("is_satisfaction","sum"),
        neutral_count=("is_neutral","sum"),
        caution_count=("is_caution","sum"),
        avg_frustration_score=("frustration_score_mask","mean"),
        avg_satisfaction_score=("satisfaction_score_mask","mean"),
        avg_neutral_score=("neutral_score_mask","mean"),
        avg_caution_score=("caution_score_mask","mean"),
        avg_emotion_score_overall=("emotion_score","mean"),
    ).reset_index()

    g["frustration_ratio"] = g.apply(lambda r: _safe_ratio(r["frustration_count"], r["total_commits_emotions"]), axis=1)
    g["satisfaction_ratio"] = g.apply(lambda r: _safe_ratio(r["satisfaction_count"], r["total_commits_emotions"]), axis=1)
    g["neutral_ratio"]      = g.apply(lambda r: _safe_ratio(r["neutral_count"], r["total_commits_emotions"]), axis=1)
    g["caution_ratio"]      = g.apply(lambda r: _safe_ratio(r["caution_count"], r["total_commits_emotions"]), axis=1)

    cols = [
        "repo","time_period","total_commits_emotions",
        "frustration_count","frustration_ratio","avg_frustration_score",
        "satisfaction_count","satisfaction_ratio","avg_satisfaction_score",
        "neutral_count","neutral_ratio","avg_neutral_score",
        "caution_count","caution_ratio","avg_caution_score",
        "avg_emotion_score_overall",
    ]
    return g[cols].sort_values(["repo","time_period"]).reset_index(drop=True)

# ----------------------------
# Wave loading
# ----------------------------
def load_wave(path: str, target_freq: str):
    if not path or not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "repo" not in df.columns or "time_period" not in df.columns: return None

    df = df.copy()
    df["repo_norm"] = _repo_norm(df["repo"])
    df["tp_norm"] = _normalize_tp_series_to_freq(df["time_period"].astype(str), target_freq)
    present = [c for c in CHURN_COLS if c in df.columns]
    keep = ["repo","time_period","repo_norm","tp_norm"] + present
    df = df[keep].dropna(subset=["repo_norm","tp_norm"]).drop_duplicates(subset=["repo_norm","tp_norm"], keep="last")
    return df if len(present) else None

# ----------------------------
# Build
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Union Wave1+Wave2 + commits metrics; churn only from waves.")
    ap.add_argument("--commits", default="commits_with_emotions_final_20k.csv", help="Commits CSV (default: commits_with_emotions_final_20k.csv)")
    ap.add_argument("--wave1", default=None, help="Wave 1 master CSV")
    ap.add_argument("--wave2", default=None, help="Wave 2 master CSV")
    ap.add_argument("--freq", default="Q", choices=["Q","M","W"], help="Q (default) / M / W")
    ap.add_argument("--out", default="project_health_master.csv")
    ap.add_argument("--pbic-score-threshold", type=float, default=0.70, help="Threshold for negative sentiment in pBIC (default 0.70)")
    args = ap.parse_args()

    # Load commits and compute Tasks 1–3
    commits = pd.read_csv(args.commits)
    need = {"repo","date","message","emotion_predicted","emotion_score"}
    miss = need - set(commits.columns)
    if miss:
        raise SystemExit(f"Missing required columns in {args.commits}: {sorted(miss)}")

    commits = add_time_period(commits, "date", args.freq)

    t1 = compute_task1_metrics(commits, pbic_score_threshold=args.pbic_score_threshold)
    t2 = compute_task2_metrics(commits)
    t3 = compute_task3_metrics(commits)
    t123 = t3.merge(t1, on=["repo","time_period"], how="outer").merge(t2, on=["repo","time_period"], how="outer")

    # Normalize keys for commits side (internal only)
    t123["repo_norm"] = _repo_norm(t123["repo"])
    t123["tp_norm"] = _normalize_tp_series_to_freq(t123["time_period"].astype(str), args.freq)

    # Load and union waves
    w1 = load_wave(args.wave1, args.freq)
    w2 = load_wave(args.wave2, args.freq)
    if w1 is not None and w2 is not None:
        waves = pd.concat([w1, w2], ignore_index=True).drop_duplicates(subset=["repo_norm","tp_norm"], keep="last")
    else:
        waves = w1 if w1 is not None else w2

    # Full key universe
    if waves is not None:
        key_union = pd.concat([
            t123[["repo","time_period","repo_norm","tp_norm"]],
            waves[["repo","time_period","repo_norm","tp_norm"]],
        ], ignore_index=True).drop_duplicates(subset=["repo_norm","tp_norm"], keep="last")
    else:
        key_union = t123[["repo","time_period","repo_norm","tp_norm"]].drop_duplicates(subset=["repo_norm","tp_norm"], keep="last")

    # Start with key_union, attach Tasks 1–3 (outer)
    master = key_union.merge(t123.drop(columns=["repo","time_period"]), on=["repo_norm","tp_norm"], how="left")

    # Prefer commits’ display repo/time_period if available
    master["repo"] = np.where(master["repo"].notna(), master["repo"], key_union["repo"])
    master["time_period"] = np.where(master["time_period"].notna(), master["time_period"], key_union["time_period"])

    # Ensure churn cols exist, then fill from waves
    for c in CHURN_COLS:
        if c not in master.columns:
            master[c] = np.nan

    if waves is not None:
        master = master.merge(
            waves[["repo_norm","tp_norm"] + [c for c in CHURN_COLS if c in waves.columns]],
            on=["repo_norm","tp_norm"], how="left", suffixes=("","__w")
        )
        for c in CHURN_COLS:
            wcol = f"{c}__w"
            if wcol in master.columns:
                master[c] = master[c].where(master[c].notna(), master[wcol])
                master.drop(columns=[wcol], inplace=True)

    # Fill missing churn with 0 (per your requirement)
    for c in CHURN_COLS:
        master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0)
    if "commits_with_churn" in master.columns:
        master["commits_with_churn"] = master["commits_with_churn"].astype(int)

    # Drop internal keys from the final CSV
    master = master.drop(columns=["repo_norm","tp_norm"], errors="ignore")

    # Order + save
    preferred = [
        "repo","time_period",
        "pBFC_rate","pRevert_rate","pBIC_rate","pBIC_count",
        "total_commits_prod","unique_authors","avg_commits_per_author",
        "total_commits_emotions",
        "frustration_count","frustration_ratio","avg_frustration_score",
        "satisfaction_count","satisfaction_ratio","avg_satisfaction_score",
        "neutral_count","neutral_ratio","avg_neutral_score",
        "caution_count","caution_ratio","avg_caution_score",
        "avg_emotion_score_overall",
        "avg_files_changed","avg_lines_added","avg_lines_deleted","commits_with_churn",
    ]
    ordered = [c for c in preferred if c in master.columns] + [c for c in master.columns if c not in preferred]
    master = master[ordered].sort_values(["repo","time_period"]).reset_index(drop=True)

    master.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
