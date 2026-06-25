from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DEFAULT_BOUT_DEFINITIONS = [
    {"prefix": "Short_Term", "introduced": "Short_Term_Introduced", "removed": "Short_Term_Removed"},
    {"prefix": "Long_Term", "introduced": "Long_Term_Introduced", "removed": "Long_Term_Removed"},
    {"prefix": "Novel", "introduced": "Novel_Introduced", "removed": "Novel_Removed"},
]


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _load_sleap_analysis_h5(h5_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]
        if tracks.ndim != 4:
            raise ValueError(f"Unexpected tracks shape in {h5_path}: {tracks.shape}")

        # SLEAP analysis export is usually (instances, dims, nodes, frames).
        if tracks.shape[1] == 2:
            coordinates = tracks.transpose((3, 2, 1, 0))
        else:
            coordinates = tracks.transpose((0, 2, 1, 3))

        track_names = [_decode_name(v) for v in f["track_names"][:]]
        node_names = [_decode_name(v) for v in f["node_names"][:]]

        if "point_scores" in f:
            point_scores = f["point_scores"][:]
            if point_scores.ndim == 3:
                if point_scores.shape[0] == len(track_names):
                    confidences = point_scores.transpose((2, 1, 0))
                else:
                    confidences = point_scores
            else:
                raise ValueError(f"Unexpected point_scores shape in {h5_path}: {point_scores.shape}")
        else:
            confidences = np.ones((coordinates.shape[0], coordinates.shape[1], coordinates.shape[3]), dtype=float)

    return coordinates.astype(float), confidences.astype(float), node_names, track_names


def _match_sleap_file(trial_name: str, subject_name: str, sleap_dir: str | Path) -> Path | None:
    sleap_dir = Path(sleap_dir)
    files = sorted(sleap_dir.glob("*.analysis.h5"))

    trial_matches = [p for p in files if trial_name in p.name]
    if trial_matches:
        return trial_matches[0]

    subject_matches = [p for p in files if subject_name in p.name]
    if subject_matches:
        subject_matches.sort(key=lambda p: (len(p.name), p.name))
        return subject_matches[0]

    return None


def _load_behavior_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Start (s)"] = pd.to_numeric(df["Start (s)"], errors="coerce")
    df["Stop (s)"] = pd.to_numeric(df["Stop (s)"], errors="coerce")
    if "Subject" in df.columns:
        df = df[df["Subject"] == "Subject"].copy()
    return df


def _extract_bouts(
    behavior_df: pd.DataFrame,
    bout_definitions: list[dict[str, str]] | None = None,
) -> list[dict[str, float | str]]:
    bout_definitions = bout_definitions or DEFAULT_BOUT_DEFINITIONS
    bouts: list[dict[str, float | str]] = []
    for bd in bout_definitions:
        starts = (
            behavior_df.loc[behavior_df["Behavior"] == bd["introduced"], "Start (s)"]
            .dropna()
            .sort_values()
            .to_numpy()
        )
        ends = (
            behavior_df.loc[behavior_df["Behavior"] == bd["removed"], "Start (s)"]
            .dropna()
            .sort_values()
            .to_numpy()
        )
        n = min(len(starts), len(ends))
        for i in range(n):
            bouts.append(
                {
                    "bout_name": f"{bd['prefix']}-{i + 1}",
                    "bout_start_s": float(starts[i]),
                    "bout_end_s": float(ends[i]),
                }
            )
    return bouts


def _build_default_skeleton(bodyparts: list[str]) -> list[list[str]]:
    bodypart_set = set(bodyparts)
    preferred_edges = [
        ["Tail_Base", "Left_Lateral"],
        ["Tail_Base", "Right_Lateral"],
        ["Tail_Base", "Center"],
        ["Center", "Right_Lateral"],
        ["Center", "Left_Lateral"],
        ["Center", "Neck"],
        ["Neck", "Right_Lateral"],
        ["Neck", "Left_Lateral"],
        ["Neck", "Head"],
        ["Head", "Nose"],
    ]
    return [edge for edge in preferred_edges if edge[0] in bodypart_set and edge[1] in bodypart_set]


def _recording_frame_table(
    recording_name: str,
    coords: np.ndarray,
    confs: np.ndarray,
    *,
    subject_name: str,
    trial_name: str,
    track_name: str,
    source_h5: str,
    fps: float,
    bout_name: str | None = None,
    bout_start_s: float | None = None,
    bout_end_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = {
        "recording_name": recording_name,
        "subject_name": subject_name,
        "trial_name": trial_name,
        "track_name": track_name,
        "source_h5": source_h5,
        "n_frames": int(coords.shape[0]),
        "fps": float(fps),
        "bout_name": bout_name,
        "bout_start_s": bout_start_s,
        "bout_end_s": bout_end_s,
    }
    return coords, confs, metadata


def build_home_cage_dataset(
    experiment_dir: str | Path,
    sleap_dir: str | Path,
    fps: float = 10.0,
    include_tracks: tuple[str, ...] = ("subject", "agent"),
    use_bouts: bool = False,
    behavior_dir: str | Path | None = None,
    bout_definitions: list[dict[str, str]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], list[list[str]], pd.DataFrame]:
    experiment_dir = Path(experiment_dir)
    sleap_dir = Path(sleap_dir)

    coordinates: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    resolved_bodyparts: list[str] | None = None
    resolved_skeleton: list[list[str]] | None = None

    trial_dirs = sorted(p for p in experiment_dir.iterdir() if p.is_dir())
    for trial_dir in trial_dirs:
        trial_name = trial_dir.name
        subject_name = trial_name.split("-")[0]

        sleap_path = _match_sleap_file(trial_name, subject_name, sleap_dir)
        if sleap_path is None:
            print(f"Warning: no SLEAP file found for {trial_name}. Skipping.")
            continue

        raw_coords, raw_confs, bodyparts, track_names = _load_sleap_analysis_h5(sleap_path)
        if resolved_bodyparts is None:
            resolved_bodyparts = bodyparts
            resolved_skeleton = _build_default_skeleton(bodyparts)

        track_lookup = {name: i for i, name in enumerate(track_names)}
        selected_tracks = [t for t in include_tracks if t in track_lookup]
        if not selected_tracks:
            print(f"Warning: none of {include_tracks} found in {sleap_path.name}. Skipping.")
            continue

        bouts: list[dict[str, float | str]] | None = None
        if use_bouts:
            if behavior_dir is None:
                raise ValueError("behavior_dir must be provided when use_bouts=True")
            behavior_path = Path(behavior_dir) / f"{trial_name}.csv"
            if not behavior_path.exists():
                print(f"Warning: no behavior CSV found for {trial_name}. Skipping bout export.")
                continue
            behavior_df = _load_behavior_csv(behavior_path)
            bouts = _extract_bouts(behavior_df, bout_definitions=bout_definitions)
            if not bouts:
                print(f"Warning: no bouts extracted for {trial_name}. Skipping.")
                continue

        for track_name in selected_tracks:
            track_ix = track_lookup[track_name]
            track_coords = raw_coords[:, :, :, track_ix]
            track_confs = raw_confs[:, :, track_ix]

            if use_bouts and bouts is not None:
                for bout in bouts:
                    start_s = float(bout["bout_start_s"])
                    end_s = float(bout["bout_end_s"])
                    start_ix = max(0, int(np.floor(start_s * fps)))
                    end_ix = min(track_coords.shape[0], int(np.floor(end_s * fps)) + 1)
                    if end_ix <= start_ix:
                        continue

                    rec_name = f"{subject_name}_{track_name}_{bout['bout_name']}"
                    rec_coords, rec_confs, rec_meta = _recording_frame_table(
                        rec_name,
                        track_coords[start_ix:end_ix].copy(),
                        track_confs[start_ix:end_ix].copy(),
                        subject_name=subject_name,
                        trial_name=trial_name,
                        track_name=track_name,
                        source_h5=str(sleap_path),
                        fps=fps,
                        bout_name=str(bout["bout_name"]),
                        bout_start_s=start_s,
                        bout_end_s=end_s,
                    )
                    coordinates[rec_name] = rec_coords
                    confidences[rec_name] = rec_confs
                    metadata_rows.append(rec_meta)
            else:
                rec_name = f"{subject_name}_{track_name}"
                rec_coords, rec_confs, rec_meta = _recording_frame_table(
                    rec_name,
                    track_coords.copy(),
                    track_confs.copy(),
                    subject_name=subject_name,
                    trial_name=trial_name,
                    track_name=track_name,
                    source_h5=str(sleap_path),
                    fps=fps,
                )
                coordinates[rec_name] = rec_coords
                confidences[rec_name] = rec_confs
                metadata_rows.append(rec_meta)

    metadata_df = pd.DataFrame(metadata_rows)
    if not metadata_df.empty:
        metadata_df = metadata_df.sort_values(["subject_name", "trial_name", "track_name", "recording_name"]).reset_index(drop=True)

    return coordinates, confidences, (resolved_bodyparts or []), (resolved_skeleton or []), metadata_df


def save_kpms_inputs(
    output_path: str | Path,
    coordinates: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    bodyparts: list[str],
    skeleton: list[list[str]],
    metadata_df: pd.DataFrame,
) -> None:
    payload = {
        "coordinates": coordinates,
        "confidences": confidences,
        "bodyparts": list(bodyparts),
        "skeleton": list(skeleton),
        "metadata_df": metadata_df.copy(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_kpms_inputs(input_path: str | Path) -> dict[str, Any]:
    with Path(input_path).open("rb") as f:
        return pickle.load(f)


def results_to_frame_table(results: dict[str, dict[str, Any]], metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata_index = metadata_df.set_index("recording_name", drop=False)
    rows: list[pd.DataFrame] = []

    for recording_name, rec in results.items():
        if recording_name not in metadata_index.index:
            continue

        syllable = rec.get("syllable")
        if syllable is None:
            syllable = rec.get("syllables")
        if syllable is None:
            continue

        syllable = np.asarray(syllable)
        n_frames = len(syllable)

        centroid = rec.get("centroid")
        if centroid is None:
            centroid = np.full((n_frames, 2), np.nan, dtype=float)
        else:
            centroid = np.asarray(centroid, dtype=float)
            if centroid.shape[0] != n_frames:
                centroid = centroid[:n_frames]

        heading = rec.get("heading")
        if heading is None:
            heading = np.full(n_frames, np.nan, dtype=float)
        else:
            heading = np.asarray(heading, dtype=float)[:n_frames]

        meta = metadata_index.loc[recording_name]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]

        fps = float(meta["fps"])
        frame_df = pd.DataFrame(
            {
                "recording_name": recording_name,
                "frame_index": np.arange(n_frames, dtype=int),
                "syllable": syllable.astype(int, copy=False),
                "time_s": np.arange(n_frames, dtype=float) / fps,
                "centroid_x": centroid[:, 0],
                "centroid_y": centroid[:, 1],
                "heading": heading,
            }
        )

        for col in metadata_df.columns:
            if col == "recording_name":
                continue
            frame_df[col] = meta[col]

        rows.append(frame_df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "recording_name",
                "frame_index",
                "syllable",
                "time_s",
                "centroid_x",
                "centroid_y",
                "heading",
            ]
            + [c for c in metadata_df.columns if c != "recording_name"]
        )

    out = pd.concat(rows, ignore_index=True)
    preferred_order = [
        "recording_name",
        "frame_index",
        "syllable",
        "time_s",
        "centroid_x",
        "centroid_y",
        "heading",
        "subject_name",
        "trial_name",
        "track_name",
        "source_h5",
        "n_frames",
        "fps",
        "bout_name",
        "bout_start_s",
        "bout_end_s",
    ]
    ordered_cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
    return out.loc[:, ordered_cols]


def filter_home_cage_recordings_to_valid_frames(
    coordinates: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    valid_frame_df: pd.DataFrame,
    *,
    time_col: str = "time_s",
    subject_col: str = "mouse_identity",
    intruder_col: str = "intruder_identity",
    keep_only_labeled_intruder: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    if metadata_df.empty:
        return coordinates.copy(), confidences.copy(), metadata_df.copy()

    required_cols = {time_col, subject_col}
    missing = required_cols.difference(valid_frame_df.columns)
    if missing:
        raise ValueError(f"valid_frame_df is missing required columns: {sorted(missing)}")

    valid_df = valid_frame_df.copy()
    valid_df[subject_col] = valid_df[subject_col].astype(str)
    valid_df[time_col] = pd.to_numeric(valid_df[time_col], errors="coerce")
    valid_df = valid_df.dropna(subset=[subject_col, time_col]).copy()

    if keep_only_labeled_intruder and intruder_col in valid_df.columns:
        valid_df[intruder_col] = valid_df[intruder_col].astype(str)
        valid_df = valid_df[valid_df[intruder_col].str.strip().ne("")].copy()
        valid_df = valid_df[valid_df[intruder_col].str.lower().ne("none")].copy()
        valid_df = valid_df[valid_df[intruder_col].str.lower().ne("nan")].copy()

    valid_frame_ix_by_subject: dict[str, np.ndarray] = {}
    for subject_name, part in valid_df.groupby(subject_col, sort=False):
        subject_rows = metadata_df[metadata_df["subject_name"].astype(str) == str(subject_name)]
        if subject_rows.empty:
            continue

        fps_values = subject_rows["fps"].dropna().unique()
        fps = float(fps_values[0]) if len(fps_values) else 10.0
        frame_ix = np.rint(part[time_col].to_numpy(dtype=float) * fps).astype(int)
        frame_ix = np.unique(frame_ix[frame_ix >= 0])
        valid_frame_ix_by_subject[str(subject_name)] = frame_ix

    filtered_coordinates: dict[str, np.ndarray] = {}
    filtered_confidences: dict[str, np.ndarray] = {}
    filtered_meta_rows: list[dict[str, Any]] = []

    for _, meta in metadata_df.iterrows():
        recording_name = str(meta["recording_name"])
        subject_name = str(meta["subject_name"])
        coords = coordinates.get(recording_name)
        confs = confidences.get(recording_name)
        if coords is None or confs is None:
            continue

        valid_ix = valid_frame_ix_by_subject.get(subject_name)
        if valid_ix is None or len(valid_ix) == 0:
            continue

        keep_ix = valid_ix[valid_ix < coords.shape[0]]
        if len(keep_ix) == 0:
            continue

        filtered_coordinates[recording_name] = coords[keep_ix].copy()
        filtered_confidences[recording_name] = confs[keep_ix].copy()

        updated_meta = meta.to_dict()
        updated_meta["n_frames"] = int(len(keep_ix))
        filtered_meta_rows.append(updated_meta)

    filtered_metadata_df = pd.DataFrame(filtered_meta_rows, columns=metadata_df.columns)
    if not filtered_metadata_df.empty:
        filtered_metadata_df = filtered_metadata_df.reset_index(drop=True)

    return filtered_coordinates, filtered_confidences, filtered_metadata_df


def _require_columns(frame_df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in frame_df.columns]
    if missing:
        raise ValueError(f"frame_df is missing required columns: {missing}")


def _entropy_from_counts(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def _distribution_from_counts(counts: pd.Series, labels: np.ndarray) -> np.ndarray:
    if counts.empty:
        return np.zeros(len(labels), dtype=float)
    aligned = counts.reindex(labels, fill_value=0).to_numpy(dtype=float)
    total = aligned.sum()
    if total <= 0:
        return np.zeros(len(labels), dtype=float)
    return aligned / total


def _jensen_shannon_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 and q_sum <= 0:
        return 1.0
    if p_sum <= 0 or q_sum <= 0:
        return 0.0

    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def _kl_div(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js_div = 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)
    js_div = min(max(js_div, 0.0), 1.0)
    return 1.0 - js_div


def extract_syllable_runs(
    frame_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    recording_col: str = "recording_name",
    frame_col: str = "frame_index",
    time_col: str = "time_s",
) -> pd.DataFrame:
    _require_columns(frame_df, [recording_col, syllable_col, frame_col])

    work = frame_df.copy()
    work[syllable_col] = pd.to_numeric(work[syllable_col], errors="coerce")
    work = work.dropna(subset=[syllable_col]).copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                recording_col,
                "run_id",
                syllable_col,
                "start_frame",
                "end_frame",
                "run_length_frames",
                "start_time_s",
                "end_time_s",
                "run_length_s",
            ]
        )

    work[syllable_col] = work[syllable_col].astype(int)
    work = work.sort_values([recording_col, frame_col]).reset_index(drop=True)
    is_new_run = (
        work[recording_col].ne(work[recording_col].shift())
        | work[syllable_col].ne(work[syllable_col].shift())
    )
    work["run_id"] = is_new_run.cumsum().astype(int)

    group_cols = [recording_col, "run_id", syllable_col]
    runs = (
        work.groupby(group_cols, as_index=False, sort=False)
        .agg(
            start_frame=(frame_col, "min"),
            end_frame=(frame_col, "max"),
        )
    )
    runs["run_length_frames"] = runs["end_frame"] - runs["start_frame"] + 1

    if time_col in work.columns:
        time_summary = (
            work.groupby(group_cols, as_index=False, sort=False)
            .agg(
                start_time_s=(time_col, "min"),
                end_time_s=(time_col, "max"),
            )
        )
        runs = runs.merge(time_summary, on=group_cols, how="left")

        fps_by_recording = None
        if "fps" in work.columns:
            fps_by_recording = work.groupby(recording_col, sort=False)["fps"].first()

        runs["run_length_s"] = runs["end_time_s"] - runs["start_time_s"]
        if fps_by_recording is not None:
            fps_map = fps_by_recording.to_dict()
            runs["run_length_s"] = (
                runs["run_length_frames"] / runs[recording_col].map(fps_map).replace(0, np.nan)
            )
    else:
        runs["start_time_s"] = np.nan
        runs["end_time_s"] = np.nan
        runs["run_length_s"] = np.nan

    passthrough_cols = [
        col
        for col in ["subject_name", "trial_name", "track_name", "bout_name", "fps"]
        if col in work.columns
    ]
    if passthrough_cols:
        meta = work.groupby(recording_col, sort=False)[passthrough_cols].first().reset_index()
        runs = runs.merge(meta, on=recording_col, how="left")

    return runs


def build_syllable_transition_table(
    frame_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    recording_col: str = "recording_name",
    frame_col: str = "frame_index",
) -> pd.DataFrame:
    runs = extract_syllable_runs(
        frame_df,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
    )
    if runs.empty:
        return pd.DataFrame(
            columns=[
                recording_col,
                "from_syllable",
                "to_syllable",
                "from_run_id",
                "to_run_id",
            ]
        )

    runs = runs.sort_values([recording_col, "start_frame"]).reset_index(drop=True)
    next_runs = runs.groupby(recording_col, sort=False).shift(-1)
    transitions = runs[[recording_col, "run_id", syllable_col]].copy()
    transitions["to_run_id"] = next_runs["run_id"]
    transitions["to_syllable"] = next_runs[syllable_col]
    transitions = transitions.dropna(subset=["to_run_id", "to_syllable"]).copy()
    transitions = transitions.rename(
        columns={
            "run_id": "from_run_id",
            syllable_col: "from_syllable",
        }
    )
    transitions["from_syllable"] = transitions["from_syllable"].astype(int)
    transitions["to_syllable"] = transitions["to_syllable"].astype(int)
    transitions["to_run_id"] = transitions["to_run_id"].astype(int)
    return transitions


def summarize_syllable_diagnostics(
    frame_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    recording_col: str = "recording_name",
    frame_col: str = "frame_index",
    time_col: str = "time_s",
    min_occupancy_frac: float = 0.005,
    short_run_frames: int = 3,
) -> pd.DataFrame:
    _require_columns(frame_df, [recording_col, syllable_col, frame_col])

    work = frame_df.copy()
    work[syllable_col] = pd.to_numeric(work[syllable_col], errors="coerce")
    work = work.dropna(subset=[syllable_col]).copy()
    if work.empty:
        return pd.DataFrame()

    work[syllable_col] = work[syllable_col].astype(int)
    total_frames = len(work)
    total_recordings = work[recording_col].nunique()

    runs = extract_syllable_runs(
        work,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
        time_col=time_col,
    )
    transitions = build_syllable_transition_table(
        work,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
    )

    frame_summary = (
        work.groupby(syllable_col, as_index=False, sort=True)
        .agg(
            n_frames=(syllable_col, "size"),
            n_recordings=(recording_col, "nunique"),
        )
    )
    frame_summary["occupancy_frac"] = frame_summary["n_frames"] / float(total_frames)
    frame_summary["recording_frac"] = frame_summary["n_recordings"] / float(max(total_recordings, 1))

    run_summary = (
        runs.groupby(syllable_col, as_index=False, sort=True)
        .agg(
            n_runs=("run_id", "size"),
            mean_run_length_frames=("run_length_frames", "mean"),
            median_run_length_frames=("run_length_frames", "median"),
            mean_run_length_s=("run_length_s", "mean"),
            median_run_length_s=("run_length_s", "median"),
        )
    )

    out_counts = (
        transitions.groupby("from_syllable")["to_syllable"]
        .value_counts()
        .rename("n")
        .reset_index()
    )
    in_counts = (
        transitions.groupby("to_syllable")["from_syllable"]
        .value_counts()
        .rename("n")
        .reset_index()
    )

    outgoing = []
    for syllable, part in out_counts.groupby("from_syllable", sort=True):
        outgoing.append(
            {
                syllable_col: int(syllable),
                "n_outgoing_transitions": int(part["n"].sum()),
                "n_distinct_next": int(part["to_syllable"].nunique()),
                "outgoing_entropy": _entropy_from_counts(part.set_index("to_syllable")["n"]),
            }
        )
    incoming = []
    for syllable, part in in_counts.groupby("to_syllable", sort=True):
        incoming.append(
            {
                syllable_col: int(syllable),
                "n_incoming_transitions": int(part["n"].sum()),
                "n_distinct_prev": int(part["from_syllable"].nunique()),
                "incoming_entropy": _entropy_from_counts(part.set_index("from_syllable")["n"]),
            }
        )

    summary = frame_summary.merge(run_summary, on=syllable_col, how="left")
    summary = summary.merge(pd.DataFrame(outgoing), on=syllable_col, how="left")
    summary = summary.merge(pd.DataFrame(incoming), on=syllable_col, how="left")
    summary = summary.fillna(
        {
            "n_runs": 0,
            "mean_run_length_frames": 0.0,
            "median_run_length_frames": 0.0,
            "mean_run_length_s": 0.0,
            "median_run_length_s": 0.0,
            "n_outgoing_transitions": 0,
            "n_distinct_next": 0,
            "outgoing_entropy": 0.0,
            "n_incoming_transitions": 0,
            "n_distinct_prev": 0,
            "incoming_entropy": 0.0,
        }
    )

    summary["frames_per_run"] = summary["n_frames"] / summary["n_runs"].replace(0, np.nan)
    summary["is_low_occupancy"] = summary["occupancy_frac"] < float(min_occupancy_frac)
    summary["is_fragmented"] = summary["median_run_length_frames"] <= int(short_run_frames)
    summary["is_low_support"] = summary["n_runs"] < 10
    summary["artifact_flag_count"] = summary[
        ["is_low_occupancy", "is_fragmented", "is_low_support"]
    ].sum(axis=1)

    return summary.sort_values(
        ["artifact_flag_count", "occupancy_frac", "median_run_length_frames"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def find_redundant_syllable_pairs(
    frame_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    recording_col: str = "recording_name",
    frame_col: str = "frame_index",
    time_col: str = "time_s",
    min_frames: int = 100,
    top_n: int = 15,
) -> pd.DataFrame:
    _require_columns(frame_df, [recording_col, syllable_col, frame_col])

    work = frame_df.copy()
    work[syllable_col] = pd.to_numeric(work[syllable_col], errors="coerce")
    work = work.dropna(subset=[syllable_col]).copy()
    if work.empty:
        return pd.DataFrame()

    work[syllable_col] = work[syllable_col].astype(int)
    runs = extract_syllable_runs(
        work,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
        time_col=time_col,
    )
    transitions = build_syllable_transition_table(
        work,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
    )
    summary = summarize_syllable_diagnostics(
        work,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
        time_col=time_col,
    ).set_index(syllable_col)

    eligible = summary.index[summary["n_frames"] >= int(min_frames)].to_numpy(dtype=int)
    if len(eligible) < 2:
        return pd.DataFrame()

    out_counts = (
        transitions.groupby("from_syllable")["to_syllable"]
        .value_counts()
        .rename("n")
        .reset_index()
    )
    in_counts = (
        transitions.groupby("to_syllable")["from_syllable"]
        .value_counts()
        .rename("n")
        .reset_index()
    )
    out_lookup = {
        int(syllable): part.set_index("to_syllable")["n"]
        for syllable, part in out_counts.groupby("from_syllable", sort=True)
    }
    in_lookup = {
        int(syllable): part.set_index("from_syllable")["n"]
        for syllable, part in in_counts.groupby("to_syllable", sort=True)
    }

    pair_rows: list[dict[str, float | int]] = []
    for idx_a, syllable_a in enumerate(eligible[:-1]):
        for syllable_b in eligible[idx_a + 1 :]:
            outgoing_sim = _jensen_shannon_similarity(
                _distribution_from_counts(out_lookup.get(int(syllable_a), pd.Series(dtype=float)), eligible),
                _distribution_from_counts(out_lookup.get(int(syllable_b), pd.Series(dtype=float)), eligible),
            )
            incoming_sim = _jensen_shannon_similarity(
                _distribution_from_counts(in_lookup.get(int(syllable_a), pd.Series(dtype=float)), eligible),
                _distribution_from_counts(in_lookup.get(int(syllable_b), pd.Series(dtype=float)), eligible),
            )

            median_a = float(summary.loc[int(syllable_a), "median_run_length_frames"])
            median_b = float(summary.loc[int(syllable_b), "median_run_length_frames"])
            dwell_sim = 1.0 - min(abs(np.log((median_a + 1.0) / (median_b + 1.0))) / np.log(4.0), 1.0)

            occupancy_a = float(summary.loc[int(syllable_a), "occupancy_frac"])
            occupancy_b = float(summary.loc[int(syllable_b), "occupancy_frac"])
            occupancy_sim = 1.0 - min(abs(np.log((occupancy_a + 1e-8) / (occupancy_b + 1e-8))) / np.log(10.0), 1.0)

            similarity_score = float(
                np.mean([outgoing_sim, incoming_sim, dwell_sim, occupancy_sim])
            )
            pair_rows.append(
                {
                    "syllable_a": int(syllable_a),
                    "syllable_b": int(syllable_b),
                    "similarity_score": similarity_score,
                    "outgoing_similarity": outgoing_sim,
                    "incoming_similarity": incoming_sim,
                    "dwell_similarity": dwell_sim,
                    "occupancy_similarity": occupancy_sim,
                    "frames_a": int(summary.loc[int(syllable_a), "n_frames"]),
                    "frames_b": int(summary.loc[int(syllable_b), "n_frames"]),
                    "median_run_a": median_a,
                    "median_run_b": median_b,
                }
            )

    if not pair_rows:
        return pd.DataFrame()

    pairs = pd.DataFrame(pair_rows)
    return pairs.sort_values(
        ["similarity_score", "frames_a", "frames_b"],
        ascending=[False, False, False],
    ).head(int(top_n)).reset_index(drop=True)


def diagnose_syllable_inventory(
    frame_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    recording_col: str = "recording_name",
    frame_col: str = "frame_index",
    time_col: str = "time_s",
    min_occupancy_frac: float = 0.005,
    short_run_frames: int = 3,
    min_pair_frames: int = 100,
    top_n_pairs: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summarize_syllable_diagnostics(
        frame_df,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
        time_col=time_col,
        min_occupancy_frac=min_occupancy_frac,
        short_run_frames=short_run_frames,
    )
    redundant_pairs = find_redundant_syllable_pairs(
        frame_df,
        syllable_col=syllable_col,
        recording_col=recording_col,
        frame_col=frame_col,
        time_col=time_col,
        min_frames=min_pair_frames,
        top_n=top_n_pairs,
    )
    return summary, redundant_pairs
