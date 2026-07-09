"""
Run the shared Hab/Dishab preprocessing pipeline and save cached experiments.

This stops before notebook-specific plotting and summary tables, so the same
.pkl files can be reused across Hab/Dishab analysis notebooks without
reloading raw TDT blocks or recomputing dopamine metrics.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiment_class import Experiment
from trial_class import Trial

print(f"Using experiment_class from: {sys.modules[Experiment.__module__].__file__}", flush=True)
print(f"Using trial_class from: {sys.modules[Trial.__module__].__file__}", flush=True)


DEFAULT_NAC_EXPERIMENT_PATH = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Hab_Dishab\All\nac"
)
DEFAULT_NAC_CSV_PATH = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Hab_Dishab\All\nac_csvs"
)
DEFAULT_MPFC_EXPERIMENT_PATH = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Hab_Dishab\All\mpfc"
)
DEFAULT_MPFC_CSV_PATH = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Hab_Dishab\All\mpfc_csvs"
)
DEFAULT_NAC_OUTPUT_PATH = Path(__file__).with_name("preprocessed_hab_dishab_nac.pkl")
DEFAULT_MPFC_OUTPUT_PATH = Path(__file__).with_name("preprocessed_hab_dishab_mpfc.pkl")


BOUT_DEFINITIONS = [
    {"prefix": "s1", "introduced": "s1_Introduced", "removed": "s1_Removed"},
    {"prefix": "s2", "introduced": "s2_Introduced", "removed": "s2_Removed"},
]


def run_default_batch_process_verbose(exp):
    for index, (trial_folder, trial) in enumerate(exp.trials.items(), start=1):
        trial_start = time.perf_counter()
        total_trials = len(exp.trials)
        print(f"\n[{index}/{total_trials}] Processing {trial_folder}...", flush=True)

        steps = [
            ("downsample", lambda: trial.downsample(target_fs=100)),
            ("remove initial LED artifact", lambda: trial.remove_initial_LED_artifact(t=30)),
            ("remove final data segment", lambda: trial.remove_final_data_segment(t=10)),
            ("low-pass filter", lambda: trial.lowpass_filter(cutoff_hz=3.0)),
            ("double-exponential baseline fit", trial.basline_drift_double_exponential),
            ("IRLS motion correction", lambda: trial.motion_correction_align_channels_IRLS(IRLS_constant=1.4)),
            ("compute dF/F", trial.compute_dFF),
            ("compute z-score", lambda: trial.compute_zscore(method="standard")),
        ]

        for step_name, step_func in steps:
            step_start = time.perf_counter()
            print(f"    - {step_name}...", flush=True)
            step_func()
            elapsed = time.perf_counter() - step_start
            print(f"      done in {elapsed:.1f}s", flush=True)

        elapsed = time.perf_counter() - trial_start
        print(f"Finished {trial_folder} in {elapsed:.1f}s", flush=True)


def build_preprocessed_experiment(
    experiment_path,
    csv_path,
    first_only=False,
    use_max_length=False,
    max_bout_duration=4,
    mode="standard",
):
    hd_exp = Experiment(experiment_path, csv_path)
    run_default_batch_process_verbose(hd_exp)
    hd_exp.group_extract_manual_annotations(
        bout_definitions=BOUT_DEFINITIONS,
        first_only=first_only,
    )
    hd_exp.compute_all_da_metrics(
        use_max_length=use_max_length,
        max_bout_duration=max_bout_duration,
        mode=mode,
    )
    return hd_exp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Hab/Dishab NAc and mPFC data and save cached experiments."
    )
    parser.add_argument(
        "--region",
        choices=["all", "nac", "mpfc"],
        default="all",
        help="Which region to preprocess.",
    )
    parser.add_argument("--nac-experiment-path", default=DEFAULT_NAC_EXPERIMENT_PATH)
    parser.add_argument("--nac-csv-path", default=DEFAULT_NAC_CSV_PATH)
    parser.add_argument("--mpfc-experiment-path", default=DEFAULT_MPFC_EXPERIMENT_PATH)
    parser.add_argument("--mpfc-csv-path", default=DEFAULT_MPFC_CSV_PATH)
    parser.add_argument("--nac-output-path", default=str(DEFAULT_NAC_OUTPUT_PATH))
    parser.add_argument("--mpfc-output-path", default=str(DEFAULT_MPFC_OUTPUT_PATH))
    parser.add_argument(
        "--nac-first-only",
        action="store_true",
        help="Keep only the first annotated event per bout for NAc.",
    )
    parser.add_argument(
        "--mpfc-first-only",
        action="store_true",
        help="Keep only the first annotated event per bout for mPFC.",
    )
    parser.add_argument("--max-bout-duration", type=float, default=4)
    parser.add_argument("--mode", default="standard")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.region in ("all", "nac"):
        print("\n=== Preprocessing Hab/Dishab NAc ===", flush=True)
        nac_exp = build_preprocessed_experiment(
            experiment_path=args.nac_experiment_path,
            csv_path=args.nac_csv_path,
            first_only=args.nac_first_only,
            max_bout_duration=args.max_bout_duration,
            mode=args.mode,
        )
        nac_exp.save_preprocessed(args.nac_output_path)

    if args.region in ("all", "mpfc"):
        print("\n=== Preprocessing Hab/Dishab mPFC ===", flush=True)
        mpfc_exp = build_preprocessed_experiment(
            experiment_path=args.mpfc_experiment_path,
            csv_path=args.mpfc_csv_path,
            first_only=args.mpfc_first_only,
            max_bout_duration=args.max_bout_duration,
            mode=args.mode,
        )
        mpfc_exp.save_preprocessed(args.mpfc_output_path)


if __name__ == "__main__":
    main()
