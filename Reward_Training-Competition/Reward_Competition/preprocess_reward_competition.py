"""
Run the shared Reward Competition preprocessing pipeline and save rc_exp.

This stops before notebook-specific analysis metrics such as compute_EI_DA()
and compute_rtc_da_metrics(), so the same .pkl can be reused across RC
analysis notebooks with different windows or downstream calculations.
"""

import argparse
from pathlib import Path

from rc_extension import Reward_Competition


DEFAULT_EXPERIMENT_PATH = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Reward_Competition\combined_cohorts"
)
DEFAULT_MANUAL_SCORING_PATH = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Reward_Competition\combined_cohorts"
    r"\manual_scoring_combined.xlsx"
)
DEFAULT_HVL_PATH = (
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Reward_Competition\combined_cohorts"
    r"\HvL_comp_scoring_updated.xlsx"
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("preprocessed_reward_competition.pkl")


def build_preprocessed_experiment(
    experiment_path,
    manual_scoring_path,
    hvl_path=None,
    keep_ties=False,
    remove_subjects=True,
    remove_tangles=True,
    tangle_placeholders=True,
):
    rc_exp = Reward_Competition(
        experiment_folder_path=experiment_path,
        behavior_folder_path=None,
    )
    rc_exp.rtc_processing()

    rc_exp.read_and_merge_manual_scoring(manual_scoring_path)

    if remove_subjects:
        rc_exp.remove_specified_subjects()

    if hvl_path:
        if keep_ties:
            rc_exp.read_hvl_scoring_keep_ties(hvl_path)
        else:
            rc_exp.read_hvl_scoring(hvl_path)

    if remove_tangles:
        rc_exp.remove_tangles(placeholders=tangle_placeholders)

    rc_exp.extract_da_columns()
    rc_exp.find_first_port_entry_after_sound_cue()

    return rc_exp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Reward Competition data and save rc_exp."
    )
    parser.add_argument("--experiment-path", default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--manual-scoring-path", default=DEFAULT_MANUAL_SCORING_PATH)
    parser.add_argument(
        "--hvl-path",
        default=DEFAULT_HVL_PATH,
        help="Optional HVL scoring file. Use --skip-hvl to omit HVL data.",
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--keep-ties",
        action="store_true",
        help="Use read_hvl_scoring_keep_ties instead of read_hvl_scoring.",
    )
    parser.add_argument(
        "--skip-hvl",
        action="store_true",
        help="Do not load HVL scoring columns into the saved object.",
    )
    parser.add_argument(
        "--keep-subjects",
        action="store_true",
        help="Do not call remove_specified_subjects().",
    )
    parser.add_argument(
        "--keep-tangles",
        action="store_true",
        help="Do not call remove_tangles().",
    )
    parser.add_argument(
        "--drop-tangles",
        action="store_true",
        help="Drop tangled events instead of preserving placeholder positions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hvl_path = None if args.skip_hvl else args.hvl_path
    rc_exp = build_preprocessed_experiment(
        experiment_path=args.experiment_path,
        manual_scoring_path=args.manual_scoring_path,
        hvl_path=hvl_path,
        keep_ties=args.keep_ties,
        remove_subjects=not args.keep_subjects,
        remove_tangles=not args.keep_tangles,
        tangle_placeholders=not args.drop_tangles,
    )
    rc_exp.save_preprocessed(args.output_path)


if __name__ == "__main__":
    main()
