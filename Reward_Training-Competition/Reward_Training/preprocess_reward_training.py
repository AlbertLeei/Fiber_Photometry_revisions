"""
Run the shared Reward Training preprocessing pipeline and save rt_exp.

This stops before notebook-specific analysis metrics such as compute_EI_DA()
and compute_rtc_da_metrics(), so the same .pkl can be reused across RT
analysis notebooks with different windows or downstream calculations.
"""

import argparse
from pathlib import Path

from rt_extension import Reward_Training


DEFAULT_EXPERIMENT_ROOT = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Reward_Training\Combined"
)
DEFAULT_DAYS = (1, 10)
DEFAULT_REGIONS = ("NAc", "mPFC")
DEFAULT_OUTPUT_DIR = Path(__file__)


def default_experiment_path(day, region):
    return Path(DEFAULT_EXPERIMENT_ROOT) / f"Day_{day}" / region


def default_output_path(day, region):
    return DEFAULT_OUTPUT_DIR.with_name(
        f"preprocessed_reward_training_day{day}_{region}.pkl"
    )


def build_preprocessed_experiment(
    experiment_path,
    remove_subjects=True,
):
    experiment_path = str(experiment_path)
    rt_exp = Reward_Training(
        experiment_folder_path=experiment_path,
        behavior_folder_path=None,
    )
    rt_exp.rtc_processing()
    rt_exp.create_base_df(experiment_path)

    if remove_subjects:
        rt_exp.remove_specified_subjects()

    rt_exp.extract_da_columns()
    rt_exp.find_first_port_entry_after_sound_cue()

    return rt_exp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Reward Training data and save rt_exp."
    )
    parser.add_argument(
        "--experiment-path",
        default=None,
        help="Process a single experiment folder instead of the default day/region batch.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output .pkl path for --experiment-path. Defaults to the old single-run name.",
    )
    parser.add_argument(
        "--keep-subjects",
        action="store_true",
        help="Do not call remove_specified_subjects().",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment_path:
        output_path = args.output_path or DEFAULT_OUTPUT_DIR.with_name(
            "preprocessed_reward_training.pkl"
        )
        rt_exp = build_preprocessed_experiment(
            experiment_path=args.experiment_path,
            remove_subjects=not args.keep_subjects,
        )
        rt_exp.save_preprocessed(output_path)
        return

    for day in DEFAULT_DAYS:
        for region in DEFAULT_REGIONS:
            rt_exp = build_preprocessed_experiment(
                experiment_path=default_experiment_path(day, region),
                remove_subjects=not args.keep_subjects,
            )
            rt_exp.save_preprocessed(default_output_path(day, region))


if __name__ == "__main__":
    main()
