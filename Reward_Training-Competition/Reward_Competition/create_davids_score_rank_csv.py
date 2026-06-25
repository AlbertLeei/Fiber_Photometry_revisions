import argparse
from pathlib import Path
import sys
import types

sys.modules.setdefault("tdt", types.ModuleType("tdt"))
from rc_extension import Reward_Competition


DEFAULT_INPUT = Path(
    r"C:\Users\alber\OneDrive\Desktop\PC_Lab\Photometry\Pilot_2"
    r"\Combined_Cohorts\Reward_Competition\combined_cohorts"
    r"\manual_scoring_combined.xlsx"
)
DEFAULT_OUTPUT = Path(__file__).with_name("reward_competition_davids_score_ranks.csv")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create a CSV of reward competition ranks from David's score."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Manual scoring Excel file used for David's score calculation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for columns ID, DS, Cage, and Rank.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    reward_competition = Reward_Competition.__new__(Reward_Competition)
    ranks = reward_competition.find_ranks_using_ds(str(args.input))
    ranks = ranks.sort_values(["Cage", "Rank", "ID"]).reset_index(drop=True)
    ranks.to_csv(args.output, index=False)

    print(f"Wrote {len(ranks)} rows to {args.output}")
    print(ranks.to_string(index=False))


if __name__ == "__main__":
    main()
