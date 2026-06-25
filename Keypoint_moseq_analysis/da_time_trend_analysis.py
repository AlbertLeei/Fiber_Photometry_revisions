from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


sns.set_style("whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name in {"Pose_Tracking", "Keypoint_moseq_analysis"}:
    ROOT = PROJECT_ROOT.parent
else:
    ROOT = PROJECT_ROOT

POSE_DIR = ROOT / "Pose_Tracking"
KPMS_DIR = ROOT / "Keypoint_moseq_analysis"

SYLLABLE_PATH = POSE_DIR / "keypoint_moseq_project_home_cage" / "2026_05_09-00_56_11" / "syllable_frames.csv"
DA_PATH = POSE_DIR / "home_cage_pose_DA_all.csv"
OUTPUT_DIR = KPMS_DIR / "da_time_trend_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_TOLERANCE_S = 0.11
MIN_POINTS_PER_GROUP = 30
N_TIME_BINS = 6
TOP_N_SYLLABLES = 12
TARGET_REGION = "mPFC"


def infer_region_from_subject(subject_name: str) -> str | pd.NA:
    subject_name = str(subject_name).strip().lower()
    if subject_name.startswith("n"):
        return "NAc"
    if subject_name.startswith("p"):
        return "mPFC"
    return pd.NA


def load_da_frame(da_path: Path) -> pd.DataFrame:
    usecols = [
        "time_s",
        "brain_region",
        "mouse_identity",
        "zscore_DA",
        "intruder_identity",
        "behavior_active",
        "agent_in_subject",
        "subject_in_agent",
    ]
    df = pd.read_csv(da_path, usecols=usecols)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["zscore_DA"] = pd.to_numeric(df["zscore_DA"], errors="coerce")
    df = df.dropna(subset=["time_s", "zscore_DA"]).copy()
    df["subject_name"] = df["mouse_identity"].astype(str).str.strip()
    df["brain_region"] = df["brain_region"].astype(str).str.strip()
    df["bout_phase"] = df["intruder_identity"].astype(str).str.strip()
    df["social_proximity"] = df["agent_in_subject"].eq("Yes") | df["subject_in_agent"].eq("Yes")
    df = df.sort_values(["subject_name", "brain_region", "bout_phase", "time_s"]).reset_index(drop=True)

    df["bout_elapsed_s"] = df.groupby(["subject_name", "brain_region", "bout_phase"], sort=False)["time_s"].transform(
        lambda s: s - s.min()
    )
    return df


def load_syllable_frame(syllable_path: Path) -> pd.DataFrame:
    usecols = [
        "recording_name",
        "frame_index",
        "syllable",
        "time_s",
        "subject_name",
        "track_name",
    ]
    df = pd.read_csv(syllable_path, usecols=usecols)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["syllable"] = pd.to_numeric(df["syllable"], errors="coerce")
    df = df.dropna(subset=["time_s", "syllable"]).copy()
    df["syllable"] = df["syllable"].astype(int)
    df["subject_name"] = df["subject_name"].astype(str).str.strip()
    df["brain_region"] = df["subject_name"].map(infer_region_from_subject)
    df = df[df["track_name"].astype(str).str.lower() == "subject"].copy()
    df = df.dropna(subset=["brain_region"]).copy()
    df = df.sort_values(["subject_name", "brain_region", "time_s"]).reset_index(drop=True)
    return df


def merge_syllables_onto_da(
    da_df: pd.DataFrame,
    syllable_df: pd.DataFrame,
    *,
    tolerance_s: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    syllable_cols = ["time_s", "syllable", "recording_name", "frame_index"]

    for (subject_name, brain_region), da_rec in da_df.groupby(["subject_name", "brain_region"], sort=False):
        da_rec = da_rec.sort_values("time_s").reset_index(drop=True)
        syll_rec = syllable_df[
            (syllable_df["subject_name"] == subject_name)
            & (syllable_df["brain_region"] == brain_region)
        ][syllable_cols].sort_values("time_s").reset_index(drop=True)

        if syll_rec.empty:
            merged = da_rec.copy()
            merged["syllable"] = pd.NA
            merged["recording_name"] = pd.NA
            merged["frame_index"] = pd.NA
            parts.append(merged)
            continue

        merged = pd.merge_asof(
            da_rec,
            syll_rec,
            on="time_s",
            direction="nearest",
            tolerance=tolerance_s,
        )
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True)
    return out


def add_time_bins(df: pd.DataFrame, group_cols: list[str], n_bins: int) -> pd.DataFrame:
    out = df.copy()
    out["time_bin"] = pd.NA
    out["time_frac"] = np.nan

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        rec = out.loc[idx].sort_values("bout_elapsed_s")
        n = len(rec)
        if n == 0:
            continue
        ranks = np.arange(n, dtype=float)
        frac = np.zeros(n, dtype=float) if n == 1 else ranks / (n - 1)
        bins = np.minimum((frac * n_bins).astype(int), n_bins - 1)
        out.loc[rec.index, "time_frac"] = frac
        out.loc[rec.index, "time_bin"] = bins

    out["time_bin"] = pd.to_numeric(out["time_bin"], errors="coerce").astype("Int64")
    return out


def compute_group_trends(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    time_col: str = "bout_elapsed_s",
    value_col: str = "zscore_DA",
    min_points: int = MIN_POINTS_PER_GROUP,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for keys, rec in df.groupby(group_cols, sort=False, dropna=False):
        rec = rec[[time_col, value_col]].dropna().sort_values(time_col)
        if len(rec) < min_points:
            continue

        x = rec[time_col].to_numpy(dtype=float)
        y = rec[value_col].to_numpy(dtype=float)
        if np.allclose(x, x[0]):
            continue

        lr = stats.linregress(x, y)
        rho, rho_p = stats.spearmanr(x, y)

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "n": int(len(rec)),
                "time_start_s": float(x.min()),
                "time_end_s": float(x.max()),
                "slope_per_s": float(lr.slope),
                "slope_per_min": float(lr.slope * 60.0),
                "intercept": float(lr.intercept),
                "r_value": float(lr.rvalue),
                "r_squared": float(lr.rvalue ** 2),
                "p_value": float(lr.pvalue),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "mean_DA": float(np.mean(y)),
                "start_DA_est": float(lr.intercept + lr.slope * x.min()),
                "end_DA_est": float(lr.intercept + lr.slope * x.max()),
                "delta_DA_est": float(lr.slope * (x.max() - x.min())),
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def summarize_binned_da(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.dropna(subset=["time_bin"])
        .groupby(["brain_region", "bout_phase", "time_bin"], as_index=False, observed=True)
        .agg(
            mean_DA=("zscore_DA", "mean"),
            median_DA=("zscore_DA", "median"),
            std_DA=("zscore_DA", "std"),
            n=("zscore_DA", "size"),
            n_subjects=("subject_name", "nunique"),
        )
    )
    out["sem_DA"] = out["std_DA"] / np.sqrt(out["n"].clip(lower=1))
    return out


def summarize_syllable_occupancy(df: pd.DataFrame, top_n_syllables: int) -> pd.DataFrame:
    valid = df.dropna(subset=["syllable", "time_bin"]).copy()
    if valid.empty:
        return pd.DataFrame()

    top_syllables = (
        valid.groupby("syllable", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(top_n_syllables)
        .index
        .tolist()
    )
    valid = valid[valid["syllable"].isin(top_syllables)].copy()

    counts = (
        valid.groupby(["brain_region", "bout_phase", "time_bin", "syllable"], as_index=False, observed=True)
        .size()
        .rename(columns={"size": "n_frames"})
    )
    totals = (
        valid.groupby(["brain_region", "bout_phase", "time_bin"], as_index=False, observed=True)
        .size()
        .rename(columns={"size": "total_frames"})
    )
    out = counts.merge(totals, on=["brain_region", "bout_phase", "time_bin"], how="left")
    out["occupancy_frac"] = out["n_frames"] / out["total_frames"]
    return out.sort_values(["brain_region", "bout_phase", "syllable", "time_bin"]).reset_index(drop=True)


def compute_within_syllable_trends(df: pd.DataFrame, top_n_syllables: int) -> pd.DataFrame:
    valid = df.dropna(subset=["syllable"]).copy()
    if valid.empty:
        return pd.DataFrame()

    top_syllables = (
        valid.groupby("syllable", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(top_n_syllables)
        .index
        .tolist()
    )
    valid = valid[valid["syllable"].isin(top_syllables)].copy()
    return compute_group_trends(
        valid,
        ["subject_name", "brain_region", "bout_phase", "syllable"],
        min_points=MIN_POINTS_PER_GROUP,
    )


def plot_group_trends(trend_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if trend_df.empty:
        return

    order = (
        trend_df.groupby("bout_phase", observed=True)["slope_per_min"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=trend_df, x="bout_phase", y="slope_per_min", order=order, color="lightsteelblue", fliersize=0)
    sns.stripplot(
        data=trend_df,
        x="bout_phase",
        y="slope_per_min",
        order=order,
        hue="subject_name",
        dodge=False,
        alpha=0.8,
        size=5,
    )
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.ylabel("DA slope per min")
    plt.xlabel("Bout phase")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_binned_da(binned_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if binned_df.empty:
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=binned_df,
        x="time_bin",
        y="mean_DA",
        hue="bout_phase",
        style="bout_phase",
        marker="o",
    )
    plt.xlabel("Time bin within bout")
    plt.ylabel("Mean z-scored DA")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_example_traces(df: pd.DataFrame, output_path: Path, region: str) -> None:
    plot_df = df[df["brain_region"] == region].copy()
    if plot_df.empty:
        return

    subjects = plot_df["subject_name"].dropna().astype(str).unique().tolist()[:6]
    plot_df = plot_df[plot_df["subject_name"].isin(subjects)].copy()
    if plot_df.empty:
        return

    g = sns.relplot(
        data=plot_df,
        x="bout_elapsed_s",
        y="zscore_DA",
        col="bout_phase",
        row="subject_name",
        kind="line",
        estimator=None,
        units="subject_name",
        alpha=0.35,
        height=2.2,
        aspect=1.4,
        facet_kws={"sharex": False, "sharey": True},
    )
    g.set_axis_labels("Seconds from bout start", "z-scored DA")
    g.figure.suptitle(f"{region} DA over time within bout", y=1.02)
    g.figure.tight_layout()
    g.figure.savefig(output_path, dpi=200)
    plt.close(g.figure)


def main() -> None:
    da_df = load_da_frame(DA_PATH)
    syllable_df = load_syllable_frame(SYLLABLE_PATH)
    merged_df = merge_syllables_onto_da(da_df, syllable_df, tolerance_s=TIME_TOLERANCE_S)
    merged_df = add_time_bins(merged_df, ["subject_name", "brain_region", "bout_phase"], N_TIME_BINS)

    merged_df.to_csv(OUTPUT_DIR / "da_with_syllables_long.csv", index=False)

    bout_trends = compute_group_trends(
        merged_df,
        ["subject_name", "brain_region", "bout_phase"],
        min_points=MIN_POINTS_PER_GROUP,
    )
    if not bout_trends.empty:
        bout_trends["is_negative_slope"] = bout_trends["slope_per_min"] < 0
        bout_trends["is_significant"] = bout_trends["p_value"] < 0.05
        bout_trends.to_csv(OUTPUT_DIR / "bout_level_da_trends.csv", index=False)

    region_summary = pd.DataFrame()
    if not bout_trends.empty:
        region_summary = (
            bout_trends.groupby(["brain_region", "bout_phase"], as_index=False, observed=True)
            .agg(
                n_subject_bouts=("subject_name", "size"),
                median_slope_per_min=("slope_per_min", "median"),
                mean_slope_per_min=("slope_per_min", "mean"),
                frac_negative=("is_negative_slope", "mean"),
                frac_significant=("is_significant", "mean"),
                mean_delta_DA_est=("delta_DA_est", "mean"),
            )
        )
        region_summary.to_csv(OUTPUT_DIR / "bout_level_da_trend_summary.csv", index=False)

    binned_da = summarize_binned_da(merged_df)
    if not binned_da.empty:
        binned_da.to_csv(OUTPUT_DIR / "binned_da_summary.csv", index=False)

    occupancy = summarize_syllable_occupancy(merged_df, TOP_N_SYLLABLES)
    if not occupancy.empty:
        occupancy.to_csv(OUTPUT_DIR / "top_syllable_occupancy_by_time_bin.csv", index=False)

    within_syllable_trends = compute_within_syllable_trends(merged_df, TOP_N_SYLLABLES)
    if not within_syllable_trends.empty:
        within_syllable_trends["is_negative_slope"] = within_syllable_trends["slope_per_min"] < 0
        within_syllable_trends["is_significant"] = within_syllable_trends["p_value"] < 0.05
        within_syllable_trends.to_csv(OUTPUT_DIR / "within_syllable_da_trends.csv", index=False)

        within_syllable_summary = (
            within_syllable_trends.groupby(["brain_region", "bout_phase", "syllable"], as_index=False, observed=True)
            .agg(
                n_subject_bouts=("subject_name", "size"),
                median_slope_per_min=("slope_per_min", "median"),
                frac_negative=("is_negative_slope", "mean"),
                frac_significant=("is_significant", "mean"),
            )
            .sort_values(["brain_region", "bout_phase", "median_slope_per_min"])
        )
        within_syllable_summary.to_csv(OUTPUT_DIR / "within_syllable_da_trend_summary.csv", index=False)

    plot_group_trends(
        bout_trends[bout_trends["brain_region"] == TARGET_REGION].copy() if not bout_trends.empty else bout_trends,
        OUTPUT_DIR / f"{TARGET_REGION.lower()}_bout_trend_slopes.png",
        f"{TARGET_REGION} DA slope by social bout",
    )
    plot_binned_da(
        binned_da[binned_da["brain_region"] == TARGET_REGION].copy() if not binned_da.empty else binned_da,
        OUTPUT_DIR / f"{TARGET_REGION.lower()}_binned_da.png",
        f"{TARGET_REGION} DA over normalized time",
    )
    plot_example_traces(merged_df, OUTPUT_DIR / f"{TARGET_REGION.lower()}_example_traces.png", TARGET_REGION)

    print(f"Wrote outputs to: {OUTPUT_DIR}")
    if not region_summary.empty:
        print(region_summary[region_summary["brain_region"] == TARGET_REGION].to_string(index=False))


if __name__ == "__main__":
    main()
