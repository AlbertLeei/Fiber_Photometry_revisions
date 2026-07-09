"""
Build a single multipanel reward-training/reward-competition summary figure.

The figure has four analysis rows per brain region:
1. Reward training Day 1 vs Day 10: Tone PSTH, Reward/PE PSTH, Tone mean DA bar,
   Reward/PE mean DA bar.
2. Reward competition win vs loss: Tone PSTH, Reward/PE PSTH, matching bars.
3. Reward competition high vs low split within wins and losses: PSTHs plus bars.
4. Consecutive outcome analysis: win-win, win-loss, loss-win, loss-loss PSTHs
   plus tone and reward consecutive-history bars.

Outputs are saved as PNG, SVG, and PDF by default.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


DARK_GRAY = "#5F6368"
BAR_WIDTH = 0.34
BAR_X_SPACING = 0.62
SUBJECT_DOT_ALPHA = 0.35
SUBJECT_DOT_SIZE = 10
BAR_EDGE_WIDTH = 1.0
SIGNIFICANCE_ALPHA = 0.05

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RT_DIR = BASE_DIR / "Reward_Training"
RC_DIR = BASE_DIR / "Reward_Competition"

for path in (PROJECT_ROOT, BASE_DIR, RT_DIR, RC_DIR):
    sys.path.insert(0, str(path))

from figure_settings import apply_plot_style, save_figure  # noqa: E402
from rt_extension import Reward_Training  # noqa: E402
from rc_extension import Reward_Competition  # noqa: E402


DEFAULT_RT_PATHS = {
    ("Day 1", "NAc"): [
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Training\Combined\Day_1\NAc",
        r"D:\PadillaCoreanoLab\Reward_Training\Combined\Day_1\NAc",
    ],
    ("Day 10", "NAc"): [
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Training\Combined\Day10\NAc",
        r"D:\PadillaCoreanoLab\Reward_Training\Combined\Day10\NAc",
    ],
    ("Day 1", "mPFC"): [
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Training\Combined\Day_1\mPFC",
        r"D:\PadillaCoreanoLab\Reward_Training\Combined\Day_1\mPFC",
    ],
    ("Day 10", "mPFC"): [
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Training\Combined\Day10\mPFC",
        r"D:\PadillaCoreanoLab\Reward_Training\Combined\Day10\mPFC",
    ],
}

DEFAULT_RC_EXPERIMENT_PATHS = [
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Competition\combined_cohorts",
]
DEFAULT_RC_MANUAL_PATHS = [
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Competition\combined_cohorts\manual_scoring_combined.xlsx",
]
DEFAULT_RC_HVL_PATHS = [
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Competition\combined_cohorts\HvL_comp_scoring_updated.xlsx",
]

REGION_COLORS = {"NAc": "#15616F", "mPFC": "#FFAF00"}
DAY_COLORS = {
    "NAc": {"Day 1": "#0B4F4A", "Day 10": "#5DB7AA"},
    "mPFC": {"Day 1": "#B57900", "Day 10": "#FFCF4A"},
}
OUTCOME_COLORS = {"Win": "#0045A6", "Loss": "#9E3311"}
COMP_COLORS = {"Low": "#818689", "High": "#171717"}
COMP_OUTCOME_COLORS = {
    ("High", "Win"): "#002D6B",
    ("High", "Loss"): "#4A1302",
    ("Low", "Win"): "#B0D1FF",
    ("Low", "Loss"): "#FF8C7E",
}
TRANSITION_COLORS = {
    "win-win": "#0045A6",
    "win-loss": "#9E3311",
    "loss-win": "#FFAF00",
    "loss-loss": "#4A1302",
}

TRANSITION_ORDER = ("win-win", "win-loss", "loss-win", "loss-loss")

DEFAULT_RT_PICKLE_PATHS = {
    ("Day 1", "NAc"): [BASE_DIR / "preprocessed_reward_training_day1_nac.pkl"],
    ("Day 10", "NAc"): [BASE_DIR / "preprocessed_reward_training_day10_nac.pkl"],
    ("Day 1", "mPFC"): [BASE_DIR / "preprocessed_reward_training_day1_mpfc.pkl"],
    ("Day 10", "mPFC"): [BASE_DIR / "preprocessed_reward_training_day10_mpfc.pkl"],
}
DEFAULT_RC_PICKLE_PATHS = [
    BASE_DIR / "Reward_Competition" / "preprocessed_reward_competition.pkl",
]
RT_PSTH_BASELINE_WINDOW = (-4, 0)
RC_PSTH_BASELINE_WINDOW = (-4, 0)
RC_PRETRIAL_BASELINE_WINDOW = (-14, -10)


def resolve_existing_path(candidates: list[str | Path], label: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    lines = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{lines}")


def cache_path(kind: str, *parts: str) -> Path:
    safe = "_".join(part.lower().replace(" ", "").replace("/", "-") for part in parts)
    return BASE_DIR / f"preprocessed_{kind}_{safe}.pkl"


def load_or_build_rt(day: str, region: str, use_cache: bool = True) -> Reward_Training:
    cpath = cache_path("reward_training", day, region)
    if use_cache and cpath.exists():
        exp = Reward_Training.load_preprocessed(cpath)
    else:
        exp_path = resolve_existing_path(DEFAULT_RT_PATHS[(day, region)], f"reward training {day} {region}")
        exp = Reward_Training(experiment_folder_path=str(exp_path), behavior_folder_path=None)
        exp.rtc_processing()
        exp.create_base_df(str(exp_path))
        exp.remove_specified_subjects()
        exp.extract_da_columns()
        exp.find_first_port_entry_after_sound_cue()
        if use_cache:
            exp.save_preprocessed(cpath)

    exp.compute_EI_DA_PrePE(
        tone_window=(-4, 20),
        pe_window=(-4, 20),
        tone_baseline_window=RT_PSTH_BASELINE_WINDOW,
        pe_baseline_window=RT_PSTH_BASELINE_WINDOW,
    )
    exp.compute_rtc_da_metrics()
    return exp


def load_rt_pickle(day: str, region: str, pickle_paths: dict | None = None) -> Reward_Training:
    paths = (pickle_paths or DEFAULT_RT_PICKLE_PATHS)[(day, region)]
    exp = Reward_Training.load_preprocessed(resolve_existing_path(paths, f"{day} {region} reward training pickle"))
    exp.compute_EI_DA_PrePE(
        tone_window=(-4, 20),
        pe_window=(-4, 20),
        tone_baseline_window=RT_PSTH_BASELINE_WINDOW,
        pe_baseline_window=RT_PSTH_BASELINE_WINDOW,
    )
    exp.compute_rtc_da_metrics()
    return exp


def load_or_build_rc(use_cache: bool = True) -> Reward_Competition:
    cpath = BASE_DIR / "Reward_Competition" / "preprocessed_reward_competition.pkl"
    if use_cache and cpath.exists():
        exp = Reward_Competition.load_preprocessed(cpath)
    else:
        exp_path = resolve_existing_path(DEFAULT_RC_EXPERIMENT_PATHS, "reward competition experiment folder")
        manual_path = resolve_existing_path(DEFAULT_RC_MANUAL_PATHS, "reward competition manual scoring")
        hvl_path = resolve_existing_path(DEFAULT_RC_HVL_PATHS, "reward competition HVL scoring")

        exp = Reward_Competition(experiment_folder_path=str(exp_path), behavior_folder_path=None)
        exp.rtc_processing()
        exp.read_and_merge_manual_scoring(str(manual_path))
        exp.remove_specified_subjects()
        exp.read_hvl_scoring(str(hvl_path))
        exp.remove_tangles(placeholders=True)
        exp.extract_da_columns()
        exp.find_first_port_entry_after_sound_cue()
        if use_cache:
            exp.save_preprocessed(cpath)

    exp.compute_EI_DA(tone_window=(-4, 10), pe_window=(-4, 10), baseline_window=RC_PSTH_BASELINE_WINDOW)
    exp.compute_pretrial_EI_DA(pretrial_window=(-10, 0), baseline_window=RC_PRETRIAL_BASELINE_WINDOW)
    exp.compute_rtc_da_metrics(include_pretrial=True, bout_duration=4)
    exp.split_by_outcome(placeholders=False)
    exp.build_transition_dfs()
    return exp


def load_rc_pickle(pickle_paths: list[str | Path] | None = None) -> Reward_Competition:
    exp = Reward_Competition.load_preprocessed(
        resolve_existing_path(pickle_paths or DEFAULT_RC_PICKLE_PATHS, "reward competition pickle")
    )
    exp.compute_EI_DA(tone_window=(-4, 10), pe_window=(-4, 10), baseline_window=RC_PSTH_BASELINE_WINDOW)
    exp.compute_pretrial_EI_DA(pretrial_window=(-10, 0), baseline_window=RC_PRETRIAL_BASELINE_WINDOW)
    exp.compute_rtc_da_metrics(include_pretrial=True, bout_duration=4)
    exp.split_by_outcome(placeholders=False)
    exp.build_transition_dfs()
    return exp


def load_summary_pickles(
    rt_pickle_paths: dict | None = None,
    rc_pickle_paths: list[str | Path] | None = None,
):
    rt_exps = {}
    for region in ("NAc", "mPFC"):
        for day in ("Day 1", "Day 10"):
            rt_exps[(day, region)] = load_rt_pickle(day, region, rt_pickle_paths)
    rc_exp = load_rc_pickle(rc_pickle_paths)
    return rt_exps, rc_exp


def save_summary_pickles(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    rt_pickle_paths: dict | None = None,
    rc_pickle_path: str | Path | None = None,
):
    paths = rt_pickle_paths or DEFAULT_RT_PICKLE_PATHS
    saved = []
    for key, exp in rt_exps.items():
        path = Path(paths[key][0])
        exp.save_preprocessed(path)
        saved.append(path)

    rc_path = Path(rc_pickle_path or DEFAULT_RC_PICKLE_PATHS[0])
    rc_exp.save_preprocessed(rc_path)
    saved.append(rc_path)
    return saved


def region_prefix(region: str) -> str:
    return "n" if region == "NAc" else "p"


def as_event_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def valid_trace(time_axis, trace) -> bool:
    try:
        x = np.asarray(time_axis, dtype=float)
        y = np.asarray(trace, dtype=float)
    except (TypeError, ValueError):
        return False
    return x.size > 1 and y.size > 1 and not np.all(np.isnan(y))


def collect_event_records(
    df: pd.DataFrame,
    event_type: str,
    region: str,
    selector: Callable[[pd.Series, int], bool] | None = None,
):
    prefix = region_prefix(region)
    records = []
    zcol = f"{event_type}_Zscore"
    tcol = f"{event_type}_Time_Axis"

    for _, row in df.iterrows():
        subject = str(row.get("subject_name", ""))
        if not subject.startswith(prefix):
            continue
        traces = as_event_list(row.get(zcol, []))
        times = as_event_list(row.get(tcol, []))
        for idx, trace in enumerate(traces):
            if selector is not None and not selector(row, idx):
                continue
            time_axis = times[idx] if idx < len(times) else []
            if not valid_trace(time_axis, trace):
                continue
            records.append((subject, np.asarray(time_axis, dtype=float), np.asarray(trace, dtype=float)))
    return records


def make_common_time(records, time_window: tuple[float, float]) -> np.ndarray | None:
    dts = []
    for _, time_axis, _ in records:
        time_axis = time_axis[np.isfinite(time_axis)]
        if time_axis.size > 1:
            diffs = np.diff(np.sort(time_axis))
            diffs = diffs[diffs > 0]
            if diffs.size:
                dts.append(float(np.nanmedian(diffs)))
    if not dts:
        return None
    dt = min(dts)
    return np.arange(time_window[0], time_window[1] + dt / 2, dt)


def subject_mean_matrix(records, time_window: tuple[float, float]):
    if not records:
        return None, None, []

    common_t = make_common_time(records, time_window)
    if common_t is None:
        return None, None, []

    by_subject: dict[str, list[np.ndarray]] = {}
    for subject, time_axis, trace in records:
        n = min(time_axis.size, trace.size)
        x = time_axis[:n]
        y = trace[:n]
        order = np.argsort(x)
        interp = np.interp(common_t, x[order], y[order], left=np.nan, right=np.nan)
        if not np.all(np.isnan(interp)):
            by_subject.setdefault(subject, []).append(interp)

    subjects = []
    means = []
    for subject, traces in sorted(by_subject.items()):
        if not traces:
            continue
        means.append(np.nanmean(np.vstack(traces), axis=0))
        subjects.append(subject)

    if not means:
        return None, None, []
    return common_t, np.vstack(means), subjects


def plot_psth_groups(
    ax,
    groups,
    event_type: str,
    region: str,
    time_window: tuple[float, float],
    title: str,
    reward_line: bool = True,
):
    plotted = False
    for label, df, selector, color in groups:
        records = collect_event_records(df, event_type, region, selector)
        time_axis, matrix, subjects = subject_mean_matrix(records, time_window)
        if matrix is None:
            continue
        mean = np.nanmean(matrix, axis=0)
        sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0]) if matrix.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(time_axis, mean, color=color, lw=2.0, label=f"{label} (n={len(subjects)})")
        ax.fill_between(time_axis, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0)
        plotted = True

    ax.axvline(0, color="black", ls="--", lw=0.9)
    if reward_line:
        ax.axvline(4, color="#C2185B", ls="--", lw=0.9)
    ax.axhline(0, color="0.75", ls="-", lw=0.6, zorder=0)
    ax.set_xlim(time_window)
    ax.set_xticks(np.arange(time_window[0], time_window[1] + 0.1, 2))
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-scored dF/F")
    if plotted:
        ax.legend(loc="upper right", fontsize=7.2)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    style_axis(ax)


def plot_reward_psth_groups(
    ax,
    groups,
    region: str,
    pe_window: tuple[float, float],
    title: str,
    baseline_window: tuple[float, float] = (-4, 0),
    gap_width: float = 1.0,
    show_legend: bool = False,
):
    baseline_width = baseline_window[1] - baseline_window[0]
    baseline_display_start = pe_window[0] - gap_width - baseline_width
    baseline_display_end = pe_window[0] - gap_width
    break_x = pe_window[0] - gap_width / 2

    plotted = False
    for label, df, selector, color in groups:
        baseline_records = collect_event_records(df, "Tone", region, selector)
        baseline_t, baseline_matrix, baseline_subjects = subject_mean_matrix(baseline_records, baseline_window)
        if baseline_matrix is not None:
            baseline_mean = np.nanmean(baseline_matrix, axis=0)
            baseline_sem = (
                np.nanstd(baseline_matrix, axis=0, ddof=1) / np.sqrt(baseline_matrix.shape[0])
                if baseline_matrix.shape[0] > 1
                else np.zeros_like(baseline_mean)
            )
            baseline_x = baseline_t + (baseline_display_start - baseline_window[0])
            ax.plot(baseline_x, baseline_mean, color=color, lw=2.0)
            ax.fill_between(
                baseline_x,
                baseline_mean - baseline_sem,
                baseline_mean + baseline_sem,
                color=color,
                alpha=0.22,
                linewidth=0,
            )
            plotted = True
        else:
            baseline_subjects = []

        pe_records = collect_event_records(df, "PE", region, selector)
        pe_t, pe_matrix, pe_subjects = subject_mean_matrix(pe_records, pe_window)
        if pe_matrix is None:
            continue
        pe_mean = np.nanmean(pe_matrix, axis=0)
        pe_sem = np.nanstd(pe_matrix, axis=0, ddof=1) / np.sqrt(pe_matrix.shape[0]) if pe_matrix.shape[0] > 1 else np.zeros_like(pe_mean)
        subjects = sorted(set(baseline_subjects).union(pe_subjects))
        ax.plot(pe_t, pe_mean, color=color, lw=2.0, label=f"{label} (n={len(subjects)})")
        ax.fill_between(pe_t, pe_mean - pe_sem, pe_mean + pe_sem, color=color, alpha=0.22, linewidth=0)
        plotted = True

    ax.axvspan(baseline_display_start, baseline_display_end, color="0.88", alpha=0.75, zorder=-2)
    ax.axvline(break_x, color=DARK_GRAY, ls=(0, (5, 5)), lw=1.0)
    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.axhline(0, color="0.75", ls="-", lw=0.6, zorder=0)
    ax.set_xlim(baseline_display_start, pe_window[1])
    ax.set_xticks([baseline_display_start, baseline_display_end, pe_window[0], 0, pe_window[1]])
    ax.set_xticklabels(["Tone -4", "Tone", "PE -4", "PE", "PE +6"])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("z-scored dF/F")
    if plotted and show_legend:
        ax.legend(loc="upper right", fontsize=7.2)
    elif not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    style_axis(ax)


def event_value(row: pd.Series, col: str, idx: int):
    values = as_event_list(row.get(col, []))
    if idx >= len(values):
        return np.nan
    try:
        return float(values[idx])
    except (TypeError, ValueError):
        return np.nan


def comp_selector(target: int):
    def _selector(row: pd.Series, idx: int) -> bool:
        comp = event_value(row, "HVL_Comp", idx)
        return np.isfinite(comp) and int(comp) == target

    return _selector


def metric_values_by_subject(
    df: pd.DataFrame,
    region: str,
    metric_col: str,
    selector: Callable[[pd.Series, int], bool] | None = None,
) -> np.ndarray:
    prefix = region_prefix(region)
    by_subject: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        subject = str(row.get("subject_name", ""))
        if not subject.startswith(prefix):
            continue
        values = as_event_list(row.get(metric_col, []))
        for idx, value in enumerate(values):
            if selector is not None and not selector(row, idx):
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                by_subject.setdefault(subject, []).append(val)

    subject_values = [np.nanmean(vals) for vals in by_subject.values() if vals]
    return np.asarray(subject_values, dtype=float)


def metric_values_by_subject_dict(
    df: pd.DataFrame,
    region: str,
    metric_col: str,
    selector: Callable[[pd.Series, int], bool] | None = None,
) -> dict[str, float]:
    prefix = region_prefix(region)
    by_subject: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        subject = str(row.get("subject_name", ""))
        if not subject.startswith(prefix):
            continue
        values = as_event_list(row.get(metric_col, []))
        for idx, value in enumerate(values):
            if selector is not None and not selector(row, idx):
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                by_subject.setdefault(subject, []).append(val)

    return {
        subject: float(np.nanmean(vals))
        for subject, vals in by_subject.items()
        if vals and np.isfinite(np.nanmean(vals))
    }


def pairwise_welch_stats(groups, comparisons: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    from scipy.stats import ttest_ind

    if comparisons is None:
        comparisons = [
            (i, j)
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
        ]

    rows = []
    for i, j in comparisons:
        left = np.asarray(groups[i]["values"], dtype=float)
        right = np.asarray(groups[j]["values"], dtype=float)
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        t_stat, p_value = ttest_ind(left, right, nan_policy="omit", equal_var=False)
        rows.append(
            {
                "group_a": groups[i]["label"],
                "group_b": groups[j]["label"],
                "n_a": int(left.size),
                "n_b": int(right.size),
                "mean_a": float(np.nanmean(left)) if left.size else np.nan,
                "mean_b": float(np.nanmean(right)) if right.size else np.nan,
                "t": float(t_stat),
                "p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def add_holm_correction(stats_df: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    stats_df = stats_df.copy()
    if stats_df.empty or p_col not in stats_df.columns:
        stats_df["p_raw"] = []
        stats_df["p_adj"] = []
        return stats_df

    stats_df["p_raw"] = stats_df[p_col]
    p_values = stats_df[p_col].to_numpy(dtype=float)
    valid = np.isfinite(p_values)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid_indices = np.where(valid)[0]
    m = valid_indices.size
    if m:
        order = valid_indices[np.argsort(p_values[valid])]
        running_max = 0.0
        for rank, idx in enumerate(order):
            corrected = (m - rank) * p_values[idx]
            running_max = max(running_max, corrected)
            adjusted[idx] = min(running_max, 1.0)
    stats_df["p_adj"] = adjusted
    stats_df[p_col] = stats_df["p_adj"]
    return stats_df


def pairwise_subject_stats(groups, comparisons: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    from scipy.stats import ttest_rel

    if comparisons is None:
        comparisons = [
            (i, j)
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
        ]

    rows = []
    for i, j in comparisons:
        left_map = groups[i]["subject_values"]
        right_map = groups[j]["subject_values"]
        shared_subjects = sorted(set(left_map).intersection(right_map))
        left = np.asarray([left_map[subject] for subject in shared_subjects], dtype=float)
        right = np.asarray([right_map[subject] for subject in shared_subjects], dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        left = left[valid]
        right = right[valid]
        if left.size >= 2:
            t_stat, p_value = ttest_rel(left, right, nan_policy="omit")
        else:
            t_stat, p_value = np.nan, np.nan
        rows.append(
            {
                "group_a": groups[i]["label"],
                "group_b": groups[j]["label"],
                "n_a": int(left.size),
                "n_b": int(right.size),
                "mean_a": float(np.nanmean(left)) if left.size else np.nan,
                "mean_b": float(np.nanmean(right)) if right.size else np.nan,
                "t": float(t_stat),
                "p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def p_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def add_sig_brackets(ax, x, means, sems, stats_df: pd.DataFrame, values=None, max_brackets: int = 4):
    if stats_df.empty:
        return
    p_col = "p_adj" if "p_adj" in stats_df.columns else "p"
    stats_df = stats_df.loc[stats_df[p_col] < SIGNIFICANCE_ALPHA].copy()
    if stats_df.empty:
        return

    finite_tops = [
        mean + sem
        for mean, sem in zip(means, sems)
        if np.isfinite(mean) and np.isfinite(sem)
    ]
    if values is not None:
        for vals in values:
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                finite_tops.append(float(np.nanmax(vals)))
    finite_bottoms = [mean for mean in means if np.isfinite(mean)]
    if values is not None:
        for vals in values:
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                finite_bottoms.append(float(np.nanmin(vals)))
    if not finite_tops or not finite_bottoms:
        return

    y_min = min(finite_bottoms + [0])
    y_max = max(finite_tops + [0])
    span = max(y_max - y_min, 0.2)
    start = y_max + span * 0.22
    step = span * 0.22

    xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
    for level, row in enumerate(stats_df.head(max_brackets).itertuples(index=False)):
        try:
            i = xlabels.index(row.group_a)
            j = xlabels.index(row.group_b)
        except ValueError:
            continue
        y = start + step * level
        bracket_h = step * 0.22
        p_value = getattr(row, p_col)
        ax.plot([x[i], x[i], x[j], x[j]], [y, y + bracket_h, y + bracket_h, y], color="black", lw=0.65, clip_on=False)
        ax.text((x[i] + x[j]) / 2, y + bracket_h * 1.12, p_to_stars(p_value), ha="center", va="bottom", fontsize=10, clip_on=False)

    current = ax.get_ylim()
    ax.set_ylim(current[0], max(current[1], start + step * (min(len(stats_df), max_brackets) + 0.95)))


def plot_bar_groups(
    ax,
    groups,
    ylabel: str,
    title: str,
    comparisons: list[tuple[int, int]] | str | None = "all",
):
    labels = [group["label"] for group in groups]
    colors = [group.get("color", "0.5") for group in groups]
    hatches = [group.get("hatch", "") for group in groups]
    values = [np.asarray(group["values"], dtype=float) for group in groups]

    means = [np.nanmean(v) if v.size else np.nan for v in values]
    sems = [
        np.nanstd(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
        for v in values
    ]

    x = np.arange(len(groups)) * BAR_X_SPACING
    for i, (mean, sem, color, hatch) in enumerate(zip(means, sems, colors, hatches)):
        ax.bar(
            x[i],
            mean,
            width=BAR_WIDTH,
            yerr=sem,
            color=color,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            capsize=4,
            hatch=hatch,
            error_kw=dict(elinewidth=BAR_EDGE_WIDTH, capthick=BAR_EDGE_WIDTH, zorder=5),
        )
        vals = values[i]
        if vals.size:
            ax.scatter(
                np.full(vals.size, x[i]),
                vals,
                s=SUBJECT_DOT_SIZE,
                facecolors="none",
                edgecolors="gray",
                alpha=SUBJECT_DOT_ALPHA,
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
            )

    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    right_pad = 0.28 if len(groups) <= 2 else 0.32
    ax.set_xlim(x[0] - 0.28, x[-1] + right_pad)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)

    if comparisons == "all":
        comparisons = None
    stats_df = pairwise_welch_stats(groups, comparisons) if comparisons is not False else pd.DataFrame()
    if comparisons is not False:
        stats_df = add_holm_correction(stats_df)
    if comparisons is not False:
        add_sig_brackets(ax, x, means, sems, stats_df, values=values)
    return stats_df


def plot_subject_bar_groups(
    ax,
    groups,
    ylabel: str,
    title: str,
    comparisons: list[tuple[int, int]] | str | None = "all",
):
    labels = [group["label"] for group in groups]
    colors = [group.get("color", "0.5") for group in groups]
    subject_maps = [group["subject_values"] for group in groups]
    values = [np.asarray(list(subject_map.values()), dtype=float) for subject_map in subject_maps]

    means = [np.nanmean(v) if v.size else np.nan for v in values]
    sems = [
        np.nanstd(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
        for v in values
    ]

    x = np.arange(len(groups)) * BAR_X_SPACING
    for i, (mean, sem, color) in enumerate(zip(means, sems, colors)):
        ax.bar(
            x[i],
            mean,
            width=BAR_WIDTH,
            yerr=sem,
            color=color,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            capsize=4,
            error_kw=dict(elinewidth=BAR_EDGE_WIDTH, capthick=BAR_EDGE_WIDTH, zorder=5),
        )

    for i, subject_map in enumerate(subject_maps):
        vals = np.asarray(list(subject_map.values()), dtype=float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        ax.scatter(
            np.full(vals.size, x[i]),
            vals,
            facecolors="none",
            edgecolors="gray",
            s=SUBJECT_DOT_SIZE,
            alpha=SUBJECT_DOT_ALPHA,
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )

    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlim(x[0] - 0.28, x[-1] + 0.28)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)

    if comparisons == "all":
        comparisons = None
    stats_df = pairwise_subject_stats(groups, comparisons) if comparisons is not False else pd.DataFrame()
    if comparisons is not False:
        stats_df = add_holm_correction(stats_df)
    if comparisons is not False:
        add_sig_brackets(ax, x, means, sems, stats_df, values=values)
    return stats_df


def transition_metric_groups(rc_exp: Reward_Competition, region: str, metric_col: str):
    groups = []
    for transition in TRANSITION_ORDER:
        df = rc_exp.transition_dfs.get(transition, pd.DataFrame())
        groups.append(
            {
                "label": transition,
                "values": metric_values_by_subject(df, region, metric_col),
                "color": TRANSITION_COLORS[transition],
            }
        )
    return groups


def rt_vs_rc_metric_groups(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    region: str,
    metric_col: str,
):
    return [
        {
            "label": "Day 10",
            "values": metric_values_by_subject(rt_exps[("Day 10", region)].da_df, region, metric_col),
            "color": DAY_COLORS[region]["Day 10"],
        },
        {
            "label": "RC Win",
            "values": metric_values_by_subject(rc_exp.winner_df, region, metric_col),
            "color": REGION_COLORS[region],
        },
    ]


def rt_day10_vs_rc_win_subject_groups(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    region: str,
    metric_col: str,
):
    return [
        {
            "label": "Day 10",
            "subject_values": metric_values_by_subject_dict(rt_exps[("Day 10", region)].da_df, region, metric_col),
            "color": DAY_COLORS[region]["Day 10"],
        },
        {
            "label": "RC Win",
            "subject_values": metric_values_by_subject_dict(rc_exp.winner_df, region, metric_col),
            "color": REGION_COLORS[region],
        },
    ]


def event_window_values_by_subject(
    df: pd.DataFrame,
    region: str,
    event_type: str = "Tone",
    window: tuple[float, float] = (-4, 0),
    selector: Callable[[pd.Series, int], bool] | None = None,
) -> np.ndarray:
    prefix = region_prefix(region)
    by_subject: dict[str, list[float]] = {}
    tcol = f"{event_type}_Time_Axis"
    zcol = f"{event_type}_Zscore"

    for _, row in df.iterrows():
        subject = str(row.get("subject_name", ""))
        if not subject.startswith(prefix):
            continue
        times = as_event_list(row.get(tcol, []))
        traces = as_event_list(row.get(zcol, []))
        for idx, trace in enumerate(traces):
            if selector is not None and not selector(row, idx):
                continue
            if idx >= len(times) or not valid_trace(times[idx], trace):
                continue
            t = np.asarray(times[idx], dtype=float)
            y = np.asarray(trace, dtype=float)
            n = min(t.size, y.size)
            mask = (t[:n] >= window[0]) & (t[:n] <= window[1])
            if np.any(mask):
                value = np.nanmean(y[:n][mask])
                if np.isfinite(value):
                    by_subject.setdefault(subject, []).append(float(value))
    return np.asarray([np.nanmean(vals) for vals in by_subject.values() if vals], dtype=float)


def quantify_baseline_by_outcome(
    rc_exp: Reward_Competition,
    region: str = "mPFC",
    baseline_window: tuple[float, float] = (-4, 0),
    event_type: str = "Tone",
) -> pd.DataFrame:
    from scipy.stats import ttest_ind

    win = event_window_values_by_subject(rc_exp.winner_df, region, event_type, baseline_window)
    loss = event_window_values_by_subject(rc_exp.loser_df, region, event_type, baseline_window)

    t_stat, p_value = ttest_ind(win, loss, nan_policy="omit", equal_var=False)
    return pd.DataFrame(
        [
            {
                "region": region,
                "event_type": event_type,
                "baseline_window_start_s": baseline_window[0],
                "baseline_window_end_s": baseline_window[1],
                "outcome": "Win",
                "n_subjects": int(win.size),
                "mean": float(np.nanmean(win)) if win.size else np.nan,
                "sem": float(np.nanstd(win, ddof=1) / np.sqrt(win.size)) if win.size > 1 else np.nan,
                "t_win_vs_loss": float(t_stat),
                "p_win_vs_loss": float(p_value),
            },
            {
                "region": region,
                "event_type": event_type,
                "baseline_window_start_s": baseline_window[0],
                "baseline_window_end_s": baseline_window[1],
                "outcome": "Loss",
                "n_subjects": int(loss.size),
                "mean": float(np.nanmean(loss)) if loss.size else np.nan,
                "sem": float(np.nanstd(loss, ddof=1) / np.sqrt(loss.size)) if loss.size > 1 else np.nan,
                "t_win_vs_loss": float(t_stat),
                "p_win_vs_loss": float(p_value),
            },
        ]
    )


def quantify_rt_vs_rc(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    region: str,
    metric_cols: tuple[str, ...] = ("Tone Mean Z-score", "PE Mean Z-score"),
) -> pd.DataFrame:
    rows = []
    for metric_col in metric_cols:
        groups = rt_vs_rc_metric_groups(rt_exps, rc_exp, region, metric_col)
        stats = pairwise_welch_stats(groups, comparisons=[(0, 1)])
        stats = add_holm_correction(stats)
        stats.insert(0, "region", region)
        stats.insert(1, "metric", metric_col)
        rows.append(stats)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bar_plot_groups_for_region(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    region: str,
) -> dict[str, list[dict]]:
    win_df = rc_exp.winner_df
    loss_df = rc_exp.loser_df
    four_groups = [
        ("Low Win", win_df, comp_selector(0), COMP_OUTCOME_COLORS[("Low", "Win")], ""),
        ("High Win", win_df, comp_selector(1), COMP_OUTCOME_COLORS[("High", "Win")], ""),
        ("Low Loss", loss_df, comp_selector(0), COMP_OUTCOME_COLORS[("Low", "Loss")], ""),
        ("High Loss", loss_df, comp_selector(1), COMP_OUTCOME_COLORS[("High", "Loss")], ""),
    ]
    return {
        "RT Tone Mean DA": [
            {"label": "Day 1", "values": metric_values_by_subject(rt_exps[("Day 1", region)].da_df, region, "Tone Mean Z-score"), "color": DAY_COLORS[region]["Day 1"]},
            {"label": "Day 10", "values": metric_values_by_subject(rt_exps[("Day 10", region)].da_df, region, "Tone Mean Z-score"), "color": DAY_COLORS[region]["Day 10"]},
        ],
        "RT Reward Mean DA": [
            {"label": "Day 1", "values": metric_values_by_subject(rt_exps[("Day 1", region)].da_df, region, "PE Mean Z-score"), "color": DAY_COLORS[region]["Day 1"]},
            {"label": "Day 10", "values": metric_values_by_subject(rt_exps[("Day 10", region)].da_df, region, "PE Mean Z-score"), "color": DAY_COLORS[region]["Day 10"]},
        ],
        "Tone: Day 10 vs RC Win": rt_vs_rc_metric_groups(rt_exps, rc_exp, region, "Tone Mean Z-score"),
        "Reward: Day 10 vs RC Win": rt_vs_rc_metric_groups(rt_exps, rc_exp, region, "PE Mean Z-score"),
        "RC Tone Mean DA": [
            {"label": "Win", "values": metric_values_by_subject(win_df, region, "Tone Mean Z-score"), "color": OUTCOME_COLORS["Win"]},
            {"label": "Loss", "values": metric_values_by_subject(loss_df, region, "Tone Mean Z-score"), "color": OUTCOME_COLORS["Loss"]},
        ],
        "RC Reward Mean DA": [
            {"label": "Win", "values": metric_values_by_subject(win_df, region, "PE Mean Z-score"), "color": OUTCOME_COLORS["Win"]},
            {"label": "Loss", "values": metric_values_by_subject(loss_df, region, "PE Mean Z-score"), "color": OUTCOME_COLORS["Loss"]},
        ],
        "RC Tone Mean DA by Comp": [
            {"label": label, "values": metric_values_by_subject(df, region, "Tone Mean Z-score", selector), "color": color, "hatch": hatch}
            for label, df, selector, color, hatch in four_groups
        ],
        "RC Reward Mean DA by Comp": [
            {"label": label, "values": metric_values_by_subject(df, region, "PE Mean Z-score", selector), "color": color, "hatch": hatch}
            for label, df, selector, color, hatch in four_groups
        ],
        "Consecutive Tone Mean DA": transition_metric_groups(rc_exp, region, "Tone Mean Z-score"),
        "Consecutive Reward Mean DA": transition_metric_groups(rc_exp, region, "PE Mean Z-score"),
    }


def quantify_all_bar_plots(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    regions: tuple[str, ...] = ("NAc", "mPFC"),
) -> pd.DataFrame:
    rows = []
    for region in regions:
        for panel, groups in bar_plot_groups_for_region(rt_exps, rc_exp, region).items():
            stats = pairwise_welch_stats(groups)
            stats = add_holm_correction(stats)
            stats.insert(0, "region", region)
            stats.insert(1, "panel", panel)
            rows.append(stats)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3.2, labelsize=8)
    ax.title.set_fontsize(9.5)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.7)


def add_region_label(fig, region_idx: int, region: str):
    y = 0.982 if region_idx == 0 else 0.506
    fig.text(0.012, y, region, fontsize=15, fontweight="bold", ha="left", va="top")


def sync_y_limits(axes, pad_fraction: float = 0.06):
    ymins = []
    ymaxs = []
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymin) and np.isfinite(ymax):
            ymins.append(ymin)
            ymaxs.append(ymax)
    if not ymins:
        return
    ymin = min(ymins)
    ymax = max(ymaxs)
    span = max(ymax - ymin, 0.1)
    ax_min = ymin - span * pad_fraction
    ax_max = ymax + span * pad_fraction
    for ax in axes:
        ax.set_ylim(ax_min, ax_max)


def apply_reward_summary_style():
    apply_plot_style(
        font_size=9,
        title_size=9.5,
        label_size=9,
        tick_size=8,
        legend_size=7.2,
        panel_width=2.4,
        panel_height=1.65,
        axis_line_width=0.7,
        tick_width=0.7,
        line_width=1.8,
        marker_size=3,
        transparent=True,
    )
    plt.rcParams["figure.constrained_layout.use"] = False


def _xlims_or_default(xlims: dict[str, tuple[float, float]] | None):
    if xlims is None:
        xlims = {}
    return {
        "rt_tone": xlims.get("rt_tone", (-4, 10)),
        "rt_reward": xlims.get("rt_reward", (-4, 6)),
        "rc_tone": xlims.get("rc_tone", (-4, 10)),
        "rc_reward": xlims.get("rc_reward", (-4, 6)),
        "comp_tone": xlims.get("comp_tone", (-4, 10)),
        "transition": xlims.get("transition", (-4, 10)),
    }


def populate_region_summary_axes(
    fig,
    outer,
    row0: int,
    region: str,
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    xlim_map: dict[str, tuple[float, float]],
):
    def compact_bar_axis(cell):
        compact = cell.subgridspec(1, 2, width_ratios=[3, 1], wspace=0.02)
        return fig.add_subplot(compact[0, 0])

    row_psth_axes = []
    row_bar_axes = []

    ax = fig.add_subplot(outer[row0, 0])
    row_psth_axes.append(ax)
    plot_psth_groups(
        ax,
        [
            ("Day 1", rt_exps[("Day 1", region)].da_df, None, DAY_COLORS[region]["Day 1"]),
            ("Day 10", rt_exps[("Day 10", region)].da_df, None, DAY_COLORS[region]["Day 10"]),
        ],
        "Tone",
        region,
        xlim_map["rt_tone"],
        "RT Tone PSTH",
    )

    ax = fig.add_subplot(outer[row0, 1])
    row_psth_axes.append(ax)
    plot_reward_psth_groups(
        ax,
        [
            ("Day 1", rt_exps[("Day 1", region)].da_df, None, DAY_COLORS[region]["Day 1"]),
            ("Day 10", rt_exps[("Day 10", region)].da_df, None, DAY_COLORS[region]["Day 10"]),
        ],
        region,
        xlim_map["rt_reward"],
        "RT Reward PSTH",
        baseline_window=RT_PSTH_BASELINE_WINDOW,
    )

    ax = fig.add_subplot(outer[row0, 2])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        [
            {"label": "Day 1", "values": metric_values_by_subject(rt_exps[("Day 1", region)].da_df, region, "Tone Mean Z-score"), "color": DAY_COLORS[region]["Day 1"]},
            {"label": "Day 10", "values": metric_values_by_subject(rt_exps[("Day 10", region)].da_df, region, "Tone Mean Z-score"), "color": DAY_COLORS[region]["Day 10"]},
        ],
        "Mean z-scored dF/F",
        "RT Tone Mean DA",
    )

    ax = fig.add_subplot(outer[row0, 3])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        [
            {"label": "Day 1", "values": metric_values_by_subject(rt_exps[("Day 1", region)].da_df, region, "PE Mean Z-score"), "color": DAY_COLORS[region]["Day 1"]},
            {"label": "Day 10", "values": metric_values_by_subject(rt_exps[("Day 10", region)].da_df, region, "PE Mean Z-score"), "color": DAY_COLORS[region]["Day 10"]},
        ],
        "Mean z-scored dF/F",
        "RT Reward Mean DA",
    )

    sync_y_limits(row_psth_axes)
    sync_y_limits(row_bar_axes)

    win_df = rc_exp.winner_df
    loss_df = rc_exp.loser_df

    row_psth_axes = []
    row_bar_axes = []
    ax = fig.add_subplot(outer[row0 + 1, 0])
    row_psth_axes.append(ax)
    plot_psth_groups(
        ax,
        [("Win", win_df, None, OUTCOME_COLORS["Win"]), ("Loss", loss_df, None, OUTCOME_COLORS["Loss"])],
        "Tone",
        region,
        xlim_map["rc_tone"],
        "RC Tone Win vs Loss",
    )

    ax = fig.add_subplot(outer[row0 + 1, 1])
    row_psth_axes.append(ax)
    plot_reward_psth_groups(
        ax,
        [("Win", win_df, None, OUTCOME_COLORS["Win"]), ("Loss", loss_df, None, OUTCOME_COLORS["Loss"])],
        region,
        xlim_map["rc_reward"],
        "RC Reward Win vs Loss",
        baseline_window=RC_PSTH_BASELINE_WINDOW,
    )

    ax = fig.add_subplot(outer[row0 + 1, 2])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        [
            {"label": "Win", "values": metric_values_by_subject(win_df, region, "Tone Mean Z-score"), "color": OUTCOME_COLORS["Win"]},
            {"label": "Loss", "values": metric_values_by_subject(loss_df, region, "Tone Mean Z-score"), "color": OUTCOME_COLORS["Loss"]},
        ],
        "Mean z-scored dF/F",
        "RC Tone Mean DA",
    )

    ax = fig.add_subplot(outer[row0 + 1, 3])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        [
            {"label": "Win", "values": metric_values_by_subject(win_df, region, "PE Mean Z-score"), "color": OUTCOME_COLORS["Win"]},
            {"label": "Loss", "values": metric_values_by_subject(loss_df, region, "PE Mean Z-score"), "color": OUTCOME_COLORS["Loss"]},
        ],
        "Mean z-scored dF/F",
        "RC Reward Mean DA",
    )

    ax = fig.add_subplot(outer[row0 + 1, 4])
    row_bar_axes.append(ax)
    plot_subject_bar_groups(
        ax,
        rt_day10_vs_rc_win_subject_groups(rt_exps, rc_exp, region, "Tone Mean Z-score"),
        "Mean z-scored dF/F",
        "Tone: Day 10 vs RC Win",
    )

    ax = fig.add_subplot(outer[row0 + 1, 5])
    row_bar_axes.append(ax)
    plot_subject_bar_groups(
        ax,
        rt_day10_vs_rc_win_subject_groups(rt_exps, rc_exp, region, "PE Mean Z-score"),
        "Mean z-scored dF/F",
        "Reward: Day 10 vs RC Win",
    )
    sync_y_limits(row_psth_axes)
    sync_y_limits(row_bar_axes)

    row_psth_axes = []
    row_bar_axes = []
    ax = fig.add_subplot(outer[row0 + 2, 0])
    row_psth_axes.append(ax)
    plot_psth_groups(
        ax,
        [
            ("Low", win_df, comp_selector(0), COMP_OUTCOME_COLORS[("Low", "Win")]),
            ("High", win_df, comp_selector(1), COMP_OUTCOME_COLORS[("High", "Win")]),
        ],
        "Tone",
        region,
        xlim_map["comp_tone"],
        "Win: Low vs High Comp",
    )

    ax = fig.add_subplot(outer[row0 + 2, 1])
    row_psth_axes.append(ax)
    plot_psth_groups(
        ax,
        [
            ("Low", loss_df, comp_selector(0), COMP_OUTCOME_COLORS[("Low", "Loss")]),
            ("High", loss_df, comp_selector(1), COMP_OUTCOME_COLORS[("High", "Loss")]),
        ],
        "Tone",
        region,
        xlim_map["comp_tone"],
        "Loss: Low vs High Comp",
    )

    four_groups = [
        ("Low Win", win_df, comp_selector(0), COMP_OUTCOME_COLORS[("Low", "Win")], ""),
        ("High Win", win_df, comp_selector(1), COMP_OUTCOME_COLORS[("High", "Win")], ""),
        ("Low Loss", loss_df, comp_selector(0), COMP_OUTCOME_COLORS[("Low", "Loss")], ""),
        ("High Loss", loss_df, comp_selector(1), COMP_OUTCOME_COLORS[("High", "Loss")], ""),
    ]
    ax = compact_bar_axis(outer[row0 + 2, 2:4])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        [
            {"label": label, "values": metric_values_by_subject(df, region, "Tone Mean Z-score", selector), "color": color, "hatch": hatch}
            for label, df, selector, color, hatch in four_groups
        ],
        "Mean z-scored dF/F",
        "RC Tone Mean DA by Comp",
    )

    ax = compact_bar_axis(outer[row0 + 2, 4:6])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        [
            {"label": label, "values": metric_values_by_subject(df, region, "PE Mean Z-score", selector), "color": color, "hatch": hatch}
            for label, df, selector, color, hatch in four_groups
        ],
        "Mean z-scored dF/F",
        "RC Reward Mean DA by Comp",
    )
    sync_y_limits(row_psth_axes)
    sync_y_limits(row_bar_axes)

    win_start_transition_groups = [
        (transition, rc_exp.transition_dfs.get(transition, pd.DataFrame()), None, TRANSITION_COLORS[transition])
        for transition in ("win-win", "win-loss")
    ]
    loss_start_transition_groups = [
        (transition, rc_exp.transition_dfs.get(transition, pd.DataFrame()), None, TRANSITION_COLORS[transition])
        for transition in ("loss-win", "loss-loss")
    ]
    row_psth_axes = []
    row_bar_axes = []
    ax = fig.add_subplot(outer[row0 + 3, 0])
    row_psth_axes.append(ax)
    plot_psth_groups(
        ax,
        win_start_transition_groups,
        "Tone",
        region,
        xlim_map["transition"],
        "Consecutive outcomes: win-win / win-loss",
    )

    ax = fig.add_subplot(outer[row0 + 3, 1])
    row_psth_axes.append(ax)
    plot_psth_groups(
        ax,
        loss_start_transition_groups,
        "Tone",
        region,
        xlim_map["transition"],
        "Consecutive outcomes: loss-win / loss-loss",
    )

    ax = compact_bar_axis(outer[row0 + 3, 2:4])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        transition_metric_groups(rc_exp, region, "Tone Mean Z-score"),
        "Mean z-scored dF/F",
        "Consecutive Tone Mean DA",
    )

    ax = compact_bar_axis(outer[row0 + 3, 4:6])
    row_bar_axes.append(ax)
    plot_bar_groups(
        ax,
        transition_metric_groups(rc_exp, region, "PE Mean Z-score"),
        "Mean z-scored dF/F",
        "Consecutive Reward Mean DA",
    )
    sync_y_limits(row_psth_axes)
    sync_y_limits(row_bar_axes)


def build_region_summary_figure(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    region: str,
    output_base: Path,
    *,
    figsize: tuple[float, float] = (12.2, 9.1),
    xlims: dict[str, tuple[float, float]] | None = None,
    save_formats: tuple[str, ...] = ("png", "svg", "pdf"),
    display: bool = False,
):
    xlim_map = _xlims_or_default(xlims)
    apply_reward_summary_style()

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    outer = fig.add_gridspec(
        4,
        6,
        width_ratios=[1, 1, 0.30, 0.30, 0.30, 0.30],
        height_ratios=[1, 1, 1, 1.08],
        left=0.065,
        right=0.985,
        top=0.94,
        bottom=0.08,
        hspace=0.78,
        wspace=0.78,
    )
    fig.suptitle(region, x=0.015, y=0.985, ha="left", va="top", fontsize=13, fontweight="bold")
    populate_region_summary_axes(fig, outer, 0, region, rt_exps, rc_exp, xlim_map)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved = save_figure(fig, output_base, formats=save_formats)
    if display:
        plt.show()
    else:
        plt.close(fig)
    return saved


def build_summary_figure(
    rt_exps: dict[tuple[str, str], Reward_Training],
    rc_exp: Reward_Competition,
    output_base: Path,
    *,
    figsize: tuple[float, float] = (12.2, 18.1),
    xlims: dict[str, tuple[float, float]] | None = None,
    save_formats: tuple[str, ...] = ("png", "svg", "pdf"),
):
    xlim_map = _xlims_or_default(xlims)
    apply_reward_summary_style()

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    outer = fig.add_gridspec(
        8,
        6,
        width_ratios=[1, 1, 0.30, 0.30, 0.30, 0.30],
        height_ratios=[1, 1, 1, 1.08, 1, 1, 1, 1.08],
        left=0.065,
        right=0.985,
        top=0.975,
        bottom=0.045,
        hspace=0.82,
        wspace=0.78,
    )

    for region_idx, region in enumerate(("NAc", "mPFC")):
        row0 = region_idx * 4
        populate_region_summary_axes(fig, outer, row0, region, rt_exps, rc_exp, xlim_map)
        add_region_label(fig, region_idx, region)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved = save_figure(fig, output_base, formats=save_formats)
    plt.close(fig)
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Create the reward summary multipanel figure.")
    parser.add_argument(
        "--output-base",
        default=str(BASE_DIR / "Reward_Summary_Figures" / "reward_training_competition_summary"),
        help="Output path without extension.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Load/save preprocessed experiment cache files to speed up repeat runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_cache = args.cache

    rt_exps = {}
    for region in ("NAc", "mPFC"):
        for day in ("Day 1", "Day 10"):
            print(f"Loading {region} reward training {day}...")
            rt_exps[(day, region)] = load_or_build_rt(day, region, use_cache=use_cache)

    print("Loading reward competition...")
    rc_exp = load_or_build_rc(use_cache=use_cache)

    saved = build_summary_figure(rt_exps, rc_exp, Path(args.output_base))
    print("Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
