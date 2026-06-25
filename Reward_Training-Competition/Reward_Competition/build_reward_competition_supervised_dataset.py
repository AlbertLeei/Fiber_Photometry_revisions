import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rc_extension import Reward_Competition


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "reward_competition_trialwise_supervised_dataset_compact.csv"

DEFAULT_PATH_CANDIDATES = {
    "experiment_path": [
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Competition\combined_cohorts",
        r"D:\PadillaCoreanoLab\Reward_Competition\combined_cohorts",
    ],
    "manual_scoring_path": [
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Competition\combined_cohorts\manual_scoring_combined.xlsx",
        r"D:\PadillaCoreanoLab\Reward_Competition\combined_cohorts\manual_scoring_combined.xlsx",
    ],
    "hvl_path": [
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Competition\combined_cohorts\HvL_comp_scoring_updated.xlsx",
        r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2\Combined_Cohorts\Reward_Competition\combined_cohorts\HvL_comp_scoring.xlsx",
        r"D:\PadillaCoreanoLab\Reward_Competition\combined_cohorts\HvL_comp_scoring_updated.xlsx",
    ],
    "csv_folder": [
        r"C:\Users\alber\Downloads\Full_data\Full_data",
    ],
}


def _shared_session_id(file_name: str) -> str | None:
    if pd.isna(file_name):
        return None
    parts = str(file_name).split("-", 1)
    return parts[1] if len(parts) > 1 else str(file_name)


def _as_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value if isinstance(value, list) else []


def _safe_list_get(seq, index, default=np.nan):
    seq = _as_list(seq)
    if index < len(seq):
        return seq[index]
    return default


def _label_outcome(winner, subject_name: str) -> str:
    if pd.isna(winner):
        return "tie"
    if isinstance(winner, str) and winner.strip().lower() == "tangle":
        return "tangle"
    return "win" if winner == subject_name else "loss"


def _binary_outcome(label: str):
    if label == "win":
        return 1.0
    if label == "loss":
        return 0.0
    return np.nan


def _coerce_trace(trace):
    arr = np.asarray(trace if isinstance(trace, (list, np.ndarray)) else [], dtype=float)
    return arr if arr.ndim == 1 else np.asarray([], dtype=float)


def _resolve_first_existing_path(candidates, label: str, required: bool = True):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    if required:
        raise FileNotFoundError(
            f"Could not find a default {label}. Checked: {list(candidates)}"
        )
    return None


def resolve_default_paths():
    return {
        "experiment_path": _resolve_first_existing_path(
            DEFAULT_PATH_CANDIDATES["experiment_path"],
            "experiment path",
        ),
        "manual_scoring_path": _resolve_first_existing_path(
            DEFAULT_PATH_CANDIDATES["manual_scoring_path"],
            "manual scoring path",
        ),
        "hvl_path": _resolve_first_existing_path(
            DEFAULT_PATH_CANDIDATES["hvl_path"],
            "HVL path",
            required=False,
        ),
        "csv_folder": _resolve_first_existing_path(
            DEFAULT_PATH_CANDIDATES["csv_folder"],
            "full-data CSV folder",
        ),
        "output_csv_path": str(DEFAULT_OUTPUT_CSV),
    }


def _coerce_time_axis(axis):
    arr = np.asarray(axis if isinstance(axis, (list, np.ndarray)) else [], dtype=float)
    return arr if arr.ndim == 1 else np.asarray([], dtype=float)


def _window_trace_metrics(trace, axis, window: tuple[float, float]):
    trace = _coerce_trace(trace)
    axis = _coerce_time_axis(axis)
    if trace.size == 0 or axis.size == 0 or trace.size != axis.size:
        return {
            "mean_z": np.nan,
            "auc": np.nan,
            "peak_z": np.nan,
            "peak_time_s": np.nan,
        }

    start_s, end_s = window
    mask = (axis >= start_s) & (axis <= end_s)
    if not np.any(mask):
        return {
            "mean_z": np.nan,
            "auc": np.nan,
            "peak_z": np.nan,
            "peak_time_s": np.nan,
        }

    seg_trace = trace[mask]
    seg_axis = axis[mask]
    if seg_trace.size == 0 or np.all(np.isnan(seg_trace)):
        return {
            "mean_z": np.nan,
            "auc": np.nan,
            "peak_z": np.nan,
            "peak_time_s": np.nan,
        }

    peak_idx = int(np.nanargmax(seg_trace))
    return {
        "mean_z": float(np.nanmean(seg_trace)),
        "auc": float(np.trapz(seg_trace, seg_axis)),
        "peak_z": float(seg_trace[peak_idx]),
        "peak_time_s": float(seg_axis[peak_idx]),
    }


def load_subject_initiated_competition_bouts(
    csv_folder: str,
    behavior_name: str = "Competition Bout",
    subject_role: str = "Subject",
    merge_touching_bouts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load full-data competition CSVs and keep only subject-initiated bouts.

    Each CSV is treated as subject-specific, so rows are keyed by the full file stem
    (for example: nn1-250204-085225) rather than by the shared session id.
    """
    csv_paths = sorted(Path(csv_folder).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {csv_folder}")

    all_rows = []

    for csv_path in csv_paths:
        try:
            bout_df = pd.read_csv(csv_path)
        except Exception:
            continue

        required_cols = {"Behavior", "Subject", "Start (s)", "Stop (s)"}
        if not required_cols.issubset(bout_df.columns):
            continue

        bout_df = bout_df.copy()
        bout_df["Start (s)"] = pd.to_numeric(bout_df["Start (s)"], errors="coerce")
        bout_df["Stop (s)"] = pd.to_numeric(bout_df["Stop (s)"], errors="coerce")

        keep_mask = (
            (bout_df["Behavior"].astype(str).str.strip() == behavior_name)
            & (bout_df["Subject"].astype(str).str.strip() == subject_role)
            & np.isfinite(bout_df["Start (s)"])
            & np.isfinite(bout_df["Stop (s)"])
            & (bout_df["Stop (s)"] > bout_df["Start (s)"])
        )
        bout_df = bout_df.loc[keep_mask].copy()
        if bout_df.empty:
            continue

        file_stem = csv_path.stem
        bout_df["file name"] = file_stem
        bout_df["subject_name"] = file_stem.split("-", 1)[0]
        bout_df["shared_session_id"] = _shared_session_id(file_stem)
        all_rows.append(bout_df)

    if not all_rows:
        raise ValueError(
            f"No subject-initiated '{behavior_name}' rows were found in {csv_folder}."
        )

    raw_bout_df = pd.concat(all_rows, ignore_index=True)

    merged_rows = []
    for file_name, group in raw_bout_df.groupby("file name"):
        intervals = (
            group[["Start (s)", "Stop (s)"]]
            .sort_values(["Start (s)", "Stop (s)"])
            .to_numpy(dtype=float)
        )
        if len(intervals) == 0:
            continue

        current_start, current_stop = intervals[0]
        merged_count = 1

        for start_s, stop_s in intervals[1:]:
            overlaps = start_s <= current_stop if merge_touching_bouts else start_s < current_stop
            if overlaps:
                current_stop = max(current_stop, stop_s)
                merged_count += 1
            else:
                merged_rows.append(
                    {
                        "file name": file_name,
                        "merged_bout_start_s": float(current_start),
                        "merged_bout_stop_s": float(current_stop),
                        "merged_bout_duration_s": float(current_stop - current_start),
                        "merged_source_rows": int(merged_count),
                    }
                )
                current_start, current_stop = start_s, stop_s
                merged_count = 1

        merged_rows.append(
            {
                "file name": file_name,
                "merged_bout_start_s": float(current_start),
                "merged_bout_stop_s": float(current_stop),
                "merged_bout_duration_s": float(current_stop - current_start),
                "merged_source_rows": int(merged_count),
            }
        )

    merged_bout_df = pd.DataFrame(merged_rows)
    if merged_bout_df.empty:
        raise ValueError("No merged subject-initiated bouts could be constructed.")

    return raw_bout_df, merged_bout_df


def _count_bouts_in_window(
    bout_group: pd.DataFrame,
    start_s: float,
    end_s: float,
    count_mode: str = "start",
) -> float:
    if bout_group.empty or not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.nan

    if count_mode == "start":
        mask = (bout_group["merged_bout_start_s"] >= start_s) & (
            bout_group["merged_bout_start_s"] < end_s
        )
    elif count_mode == "overlap":
        mask = (bout_group["merged_bout_stop_s"] > start_s) & (
            bout_group["merged_bout_start_s"] < end_s
        )
    else:
        raise ValueError("count_mode must be 'start' or 'overlap'.")

    return float(mask.sum())


def _total_bout_overlap_duration(bout_group: pd.DataFrame, start_s: float, end_s: float) -> float:
    if bout_group.empty or not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.nan

    total = 0.0
    for _, bout in bout_group.iterrows():
        overlap_start = max(start_s, float(bout["merged_bout_start_s"]))
        overlap_end = min(end_s, float(bout["merged_bout_stop_s"]))
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def build_trialwise_supervised_dataset(
    rc_exp: Reward_Competition,
    csv_folder: str,
    behavior_name: str = "Competition Bout",
    subject_role: str = "Subject",
    tone_duration_s: float = 4.0,
    pretone_summary_window: tuple[float, float] = (-10.0, 0.0),
    tone_summary_window: tuple[float, float] = (0.0, 4.0),
    reward_summary_window: tuple[float, float] = (4.0, 6.0),
    count_mode: str = "start",
    drop_last_trial_without_future: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Build one row per trial for supervised analyses.

    Output rows include:
    - compact pretone, tone, and reward DA summary values
    - current and next-trial labels
    - current and next HVL competition labels when available
    - subject-initiated competition bout count/rate during the ITI
      (tone offset to next tone onset)
    """
    if rc_exp.da_df.empty:
        raise ValueError("rc_exp.da_df is empty. Run preprocessing and extract_da_columns first.")

    if "Pretrial_Zscore" not in rc_exp.da_df.columns or "Pretrial_Time_Axis" not in rc_exp.da_df.columns:
        raise ValueError(
            "Pretone DA traces are missing. Run compute_pretrial_EI_DA(...) before exporting."
        )
    if "Tone_Zscore" not in rc_exp.da_df.columns or "Tone_Time_Axis" not in rc_exp.da_df.columns:
        raise ValueError(
            "Tone DA traces are missing. Run compute_EI_DA(...) before exporting."
        )

    _, merged_bout_df = load_subject_initiated_competition_bouts(
        csv_folder=csv_folder,
        behavior_name=behavior_name,
        subject_role=subject_role,
    )

    bout_lookup = {
        file_name: group.reset_index(drop=True)
        for file_name, group in merged_bout_df.groupby("file name")
    }

    rows = []

    for _, row in rc_exp.da_df.iterrows():
        subject_name = row["subject_name"]
        file_name = row["file name"]
        shared_id = _shared_session_id(file_name)

        cues = np.asarray(_as_list(row.get("filtered_sound_cues", [])), dtype=float)
        winners = _as_list(row.get("filtered_winner_array", []))
        precomp = _as_list(row.get("HVL_PreComp", []))
        comp = _as_list(row.get("HVL_Comp", []))
        pretone_traces = _as_list(row.get("Pretrial_Zscore", []))
        pretone_axes = _as_list(row.get("Pretrial_Time_Axis", []))
        tone_traces = _as_list(row.get("Tone_Zscore", []))
        tone_axes = _as_list(row.get("Tone_Time_Axis", []))
        bout_group = bout_lookup.get(file_name, pd.DataFrame())

        n_trials = len(cues)
        if n_trials == 0:
            continue

        last_index = n_trials - 1
        max_index = last_index if drop_last_trial_without_future else n_trials

        for trial_idx in range(max_index):
            current_cue = float(cues[trial_idx]) if trial_idx < len(cues) else np.nan
            next_cue = (
                float(cues[trial_idx + 1])
                if trial_idx + 1 < len(cues) and np.isfinite(cues[trial_idx + 1])
                else np.nan
            )

            current_winner = _safe_list_get(winners, trial_idx)
            next_winner = _safe_list_get(winners, trial_idx + 1)
            current_label = _label_outcome(current_winner, subject_name)
            next_label = _label_outcome(next_winner, subject_name) if trial_idx + 1 < len(winners) else np.nan

            pretone_trace = _coerce_trace(_safe_list_get(pretone_traces, trial_idx, default=[]))
            pretone_axis = _coerce_time_axis(_safe_list_get(pretone_axes, trial_idx, default=[]))
            tone_trace = _coerce_trace(_safe_list_get(tone_traces, trial_idx, default=[]))
            tone_axis = _coerce_time_axis(_safe_list_get(tone_axes, trial_idx, default=[]))
            pretone_metrics = _window_trace_metrics(
                pretone_trace,
                pretone_axis,
                pretone_summary_window,
            )
            tone_metrics = _window_trace_metrics(
                tone_trace,
                tone_axis,
                tone_summary_window,
            )
            reward_metrics = _window_trace_metrics(
                tone_trace,
                tone_axis,
                reward_summary_window,
            )

            has_valid_current_cue = np.isfinite(current_cue)
            has_valid_next_cue = np.isfinite(next_cue)
            tone_offset_s = current_cue + tone_duration_s if has_valid_current_cue else np.nan
            iti_start_s = tone_offset_s
            iti_end_s = next_cue
            iti_duration_s = iti_end_s - iti_start_s if has_valid_next_cue and np.isfinite(iti_start_s) else np.nan
            if np.isfinite(iti_duration_s) and iti_duration_s < 0:
                iti_duration_s = np.nan

            bout_count = _count_bouts_in_window(
                bout_group=bout_group,
                start_s=iti_start_s,
                end_s=iti_end_s,
                count_mode=count_mode,
            )
            overlap_duration_s = _total_bout_overlap_duration(
                bout_group=bout_group,
                start_s=iti_start_s,
                end_s=iti_end_s,
            )

            if np.isfinite(iti_duration_s) and iti_duration_s > 0 and np.isfinite(bout_count):
                bout_rate_hz = bout_count / iti_duration_s
                bout_rate_per_min = bout_rate_hz * 60.0
            else:
                bout_rate_hz = np.nan
                bout_rate_per_min = np.nan

            trace_is_valid = pretone_trace.size > 0 and np.isfinite(pretone_trace).any()

            row_dict = {
                "subject_name": subject_name,
                "file name": file_name,
                "shared_session_id": shared_id,
                "trial_number": trial_idx + 1,
                "next_trial_number": trial_idx + 2 if trial_idx + 1 < n_trials else np.nan,
                "current_tone_onset_s": current_cue,
                "next_tone_onset_s": next_cue,
                "tone_offset_s": tone_offset_s,
                "iti_start_s": iti_start_s,
                "iti_end_s": iti_end_s,
                "iti_duration_s": iti_duration_s,
                "pretone_mean_z": pretone_metrics["mean_z"],
                "pretone_auc": pretone_metrics["auc"],
                "pretone_peak_z": pretone_metrics["peak_z"],
                "pretone_peak_time_s": pretone_metrics["peak_time_s"],
                "tone_mean_z": tone_metrics["mean_z"],
                "tone_auc": tone_metrics["auc"],
                "tone_peak_z": tone_metrics["peak_z"],
                "tone_peak_time_s": tone_metrics["peak_time_s"],
                "reward_mean_z": reward_metrics["mean_z"],
                "reward_auc": reward_metrics["auc"],
                "reward_peak_z": reward_metrics["peak_z"],
                "reward_peak_time_s": reward_metrics["peak_time_s"],
                "current_trial_label": current_label,
                "current_trial_label_binary": _binary_outcome(current_label),
                "next_trial_label": next_label,
                "next_trial_label_binary": _binary_outcome(next_label) if isinstance(next_label, str) else np.nan,
                "current_hvl_precomp": _safe_list_get(precomp, trial_idx),
                "current_hvl_comp": _safe_list_get(comp, trial_idx),
                "next_hvl_precomp": _safe_list_get(precomp, trial_idx + 1),
                "next_hvl_comp": _safe_list_get(comp, trial_idx + 1),
                "iti_subject_comp_bout_count": bout_count,
                "iti_subject_comp_total_duration_s": overlap_duration_s,
                "iti_subject_comp_bout_rate_hz": bout_rate_hz,
                "iti_subject_comp_bout_rate_per_min": bout_rate_per_min,
                "has_valid_current_cue": bool(has_valid_current_cue),
                "has_valid_next_cue": bool(has_valid_next_cue),
                "has_valid_pretone_trace": bool(trace_is_valid),
                "has_valid_tone_trace": bool(tone_trace.size > 0 and np.isfinite(tone_trace).any()),
            }

            rows.append(row_dict)

    dataset_df = pd.DataFrame(rows)
    if dataset_df.empty:
        raise ValueError("No trial rows were generated for export.")

    metadata = {
        "n_rows": int(len(dataset_df)),
        "tone_duration_s": float(tone_duration_s),
        "pretone_summary_window_s": [float(pretone_summary_window[0]), float(pretone_summary_window[1])],
        "tone_summary_window_s": [float(tone_summary_window[0]), float(tone_summary_window[1])],
        "reward_summary_window_s": [float(reward_summary_window[0]), float(reward_summary_window[1])],
        "behavior_name": behavior_name,
        "subject_role": subject_role,
        "count_mode": count_mode,
        "drop_last_trial_without_future": bool(drop_last_trial_without_future),
    }

    return dataset_df, metadata


def prepare_reward_competition_experiment(
    experiment_path: str,
    manual_scoring_path: str,
    hvl_path: str | None = None,
    remove_specified_subjects: bool = False,
    tone_window: tuple[float, float] = (-4.0, 10.0),
    pe_window: tuple[float, float] = (-4.0, 10.0),
    ei_baseline_window: tuple[float, float] = (-20.0, -10.0),
    pretrial_window: tuple[float, float] = (-10.0, 0.0),
    pretrial_baseline_window: tuple[float, float] = (-14.0, -10.0),
) -> Reward_Competition:
    """
    Rebuild the RC preprocessing steps used in the analysis notebooks.
    """
    rc_exp = Reward_Competition(
        experiment_folder_path=experiment_path,
        behavior_folder_path=None,
    )
    rc_exp.rtc_processing()
    rc_exp.read_and_merge_manual_scoring(manual_scoring_path)

    if remove_specified_subjects:
        rc_exp.remove_specified_subjects()

    if hvl_path:
        rc_exp.read_hvl_scoring(hvl_path)

    rc_exp.remove_tangles(placeholders=True)
    rc_exp.extract_da_columns()
    rc_exp.find_first_port_entry_after_sound_cue()
    rc_exp.compute_EI_DA(
        tone_window=tone_window,
        pe_window=pe_window,
        baseline_window=ei_baseline_window,
    )
    rc_exp.compute_pretrial_EI_DA(
        pretrial_window=pretrial_window,
        baseline_window=pretrial_baseline_window,
    )
    return rc_exp


def export_trialwise_supervised_dataset(
    experiment_path: str,
    manual_scoring_path: str,
    csv_folder: str,
    output_csv_path: str,
    hvl_path: str | None = None,
    remove_specified_subjects: bool = False,
    tone_window: tuple[float, float] = (-4.0, 10.0),
    pe_window: tuple[float, float] = (-4.0, 10.0),
    ei_baseline_window: tuple[float, float] = (-20.0, -10.0),
    pretrial_window: tuple[float, float] = (-10.0, 0.0),
    pretrial_baseline_window: tuple[float, float] = (-14.0, -10.0),
    tone_duration_s: float = 4.0,
    pretone_summary_window: tuple[float, float] = (-10.0, 0.0),
    tone_summary_window: tuple[float, float] = (0.0, 4.0),
    reward_summary_window: tuple[float, float] = (4.0, 6.0),
    behavior_name: str = "Competition Bout",
    subject_role: str = "Subject",
    count_mode: str = "start",
    drop_last_trial_without_future: bool = True,
) -> tuple[pd.DataFrame, dict]:
    rc_exp = prepare_reward_competition_experiment(
        experiment_path=experiment_path,
        manual_scoring_path=manual_scoring_path,
        hvl_path=hvl_path,
        remove_specified_subjects=remove_specified_subjects,
        tone_window=tone_window,
        pe_window=pe_window,
        ei_baseline_window=ei_baseline_window,
        pretrial_window=pretrial_window,
        pretrial_baseline_window=pretrial_baseline_window,
    )

    dataset_df, metadata = build_trialwise_supervised_dataset(
        rc_exp=rc_exp,
        csv_folder=csv_folder,
        behavior_name=behavior_name,
        subject_role=subject_role,
        tone_duration_s=tone_duration_s,
        pretone_summary_window=pretone_summary_window,
        tone_summary_window=tone_summary_window,
        reward_summary_window=reward_summary_window,
        count_mode=count_mode,
        drop_last_trial_without_future=drop_last_trial_without_future,
    )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_path, index=False)

    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return dataset_df, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Reward Competition trial-wise supervised analysis data."
    )
    parser.add_argument(
        "--experiment-path",
        default=None,
        help="Path to processed RC trial folders. Defaults to your local Reward Competition combined_cohorts path.",
    )
    parser.add_argument(
        "--manual-scoring-path",
        default=None,
        help="Path to the manual scoring Excel file used by read_and_merge_manual_scoring().",
    )
    parser.add_argument(
        "--full-data-csv-folder",
        default=None,
        help="Folder containing BORIS-style full-data competition CSVs.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to reward_competition_trialwise_supervised_dataset.csv next to this script.",
    )
    parser.add_argument("--hvl-path", default=None, help="Optional HVL scoring file path.")
    parser.add_argument(
        "--remove-specified-subjects",
        action="store_true",
        help="Apply rc_exp.remove_specified_subjects() before export.",
    )
    parser.add_argument("--pretrial-start", type=float, default=-10.0)
    parser.add_argument("--pretrial-end", type=float, default=0.0)
    parser.add_argument("--pretrial-baseline-start", type=float, default=-14.0)
    parser.add_argument("--pretrial-baseline-end", type=float, default=-10.0)
    parser.add_argument("--ei-baseline-start", type=float, default=-20.0)
    parser.add_argument("--ei-baseline-end", type=float, default=-10.0)
    parser.add_argument("--tone-window-start", type=float, default=-4.0)
    parser.add_argument("--tone-window-end", type=float, default=10.0)
    parser.add_argument("--pe-window-start", type=float, default=-4.0)
    parser.add_argument("--pe-window-end", type=float, default=10.0)
    parser.add_argument("--tone-duration-s", type=float, default=4.0)
    parser.add_argument("--tone-summary-start", type=float, default=0.0)
    parser.add_argument("--tone-summary-end", type=float, default=4.0)
    parser.add_argument("--reward-summary-start", type=float, default=4.0)
    parser.add_argument("--reward-summary-end", type=float, default=6.0)
    parser.add_argument("--behavior-name", default="Competition Bout")
    parser.add_argument("--subject-role", default="Subject")
    parser.add_argument("--count-mode", choices=["start", "overlap"], default="start")
    parser.add_argument(
        "--keep-last-trial",
        action="store_true",
        help="Keep the last trial in each session even though it has no future ITI window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = resolve_default_paths()

    experiment_path = args.experiment_path or defaults["experiment_path"]
    manual_scoring_path = args.manual_scoring_path or defaults["manual_scoring_path"]
    csv_folder = args.full_data_csv_folder or defaults["csv_folder"]
    output_csv = args.output_csv or defaults["output_csv_path"]
    hvl_path = args.hvl_path if args.hvl_path is not None else defaults["hvl_path"]

    dataset_df, metadata = export_trialwise_supervised_dataset(
        experiment_path=experiment_path,
        manual_scoring_path=manual_scoring_path,
        csv_folder=csv_folder,
        output_csv_path=output_csv,
        hvl_path=hvl_path,
        remove_specified_subjects=args.remove_specified_subjects,
        tone_window=(args.tone_window_start, args.tone_window_end),
        pe_window=(args.pe_window_start, args.pe_window_end),
        ei_baseline_window=(args.ei_baseline_start, args.ei_baseline_end),
        pretrial_window=(args.pretrial_start, args.pretrial_end),
        pretrial_baseline_window=(args.pretrial_baseline_start, args.pretrial_baseline_end),
        tone_duration_s=args.tone_duration_s,
        pretone_summary_window=(args.pretrial_start, args.pretrial_end),
        tone_summary_window=(args.tone_summary_start, args.tone_summary_end),
        reward_summary_window=(args.reward_summary_start, args.reward_summary_end),
        behavior_name=args.behavior_name,
        subject_role=args.subject_role,
        count_mode=args.count_mode,
        drop_last_trial_without_future=not args.keep_last_trial,
    )
    print(f"Experiment path: {experiment_path}")
    print(f"Manual scoring path: {manual_scoring_path}")
    print(f"HVL path: {hvl_path}")
    print(f"Full-data CSV folder: {csv_folder}")
    print(f"Output CSV: {output_csv}")
    print(f"Saved {len(dataset_df)} trial rows.")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No command-line arguments supplied. Using local default Reward Competition paths.")
    main()
