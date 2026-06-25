from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "da_with_syllables_social_context_long.csv"
SUMMARY_CSV = ROOT / "syllable_mean_da_by_region_and_social_context.csv"
HEATMAP_CSV = ROOT / "syllable_mean_da_heatmap_matrix.csv"
TOP_N = 10
MIN_PAIRED_SUBJECTS = 3


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []

    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adjusted[i] = value
        prev = value

    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result.tolist()


def safe_wilcoxon(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    diff = x.to_numpy(dtype=float) - y.to_numpy(dtype=float)
    if np.allclose(diff, 0.0, equal_nan=False):
        return 0.0, 1.0
    stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(stat), float(p_value)


def get_top_syllables(df: pd.DataFrame, top_n: int) -> list[int]:
    top = (
        df["syllable"]
        .dropna()
        .astype(int)
        .value_counts()
        .head(top_n)
        .index.astype(int)
        .tolist()
    )
    return top


def build_subject_level_table(df: pd.DataFrame, top_syllables: list[int]) -> pd.DataFrame:
    work = df.copy()
    work["syllable"] = work["syllable"].astype(int)
    work = work[work["syllable"].isin(top_syllables)].copy()

    subject_level = (
        work.groupby(
            ["brain_region", "subject_name", "intruder_identity", "social_context", "syllable"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_DA=("zscore_DA", "mean"),
            n_frames=("zscore_DA", "size"),
        )
    )
    return subject_level


def run_agent_tests(subject_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    collapsed = (
        subject_level.groupby(
            ["brain_region", "subject_name", "intruder_identity", "syllable"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_DA=("mean_DA", "mean"),
            total_frames=("n_frames", "sum"),
            n_contexts=("social_context", "nunique"),
        )
    )

    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for (brain_region, syllable), group in collapsed.groupby(["brain_region", "syllable"], observed=True):
        wide = group.pivot(index="subject_name", columns="intruder_identity", values="mean_DA")
        wide = wide.sort_index(axis=1)
        complete = wide.dropna()
        agent_levels = complete.columns.tolist()

        omnibus_row: dict[str, object] = {
            "brain_region": brain_region,
            "syllable": int(syllable),
            "agent_levels": "|".join(agent_levels),
            "n_agents": len(agent_levels),
            "n_subjects_complete": int(complete.shape[0]),
        }

        if complete.shape[0] >= MIN_PAIRED_SUBJECTS and complete.shape[1] >= 3:
            stat, p_value = friedmanchisquare(*[complete[col].to_numpy() for col in complete.columns])
            kendalls_w = float(stat) / (complete.shape[0] * (complete.shape[1] - 1))
            omnibus_row.update(
                {
                    "test": "Friedman",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "kendalls_w": kendalls_w,
                }
            )
        else:
            omnibus_row.update(
                {
                    "test": "Friedman",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "kendalls_w": np.nan,
                }
            )
        omnibus_rows.append(omnibus_row)

        raw_p_values: list[float] = []
        pairwise_indices: list[int] = []
        for agent_a, agent_b in combinations(wide.columns.tolist(), 2):
            paired = wide[[agent_a, agent_b]].dropna()
            row = {
                "brain_region": brain_region,
                "syllable": int(syllable),
                "agent_a": agent_a,
                "agent_b": agent_b,
                "n_subjects_paired": int(paired.shape[0]),
            }
            if paired.shape[0] >= MIN_PAIRED_SUBJECTS:
                stat, p_value = safe_wilcoxon(paired[agent_a], paired[agent_b])
                diffs = paired[agent_a] - paired[agent_b]
                row.update(
                    {
                        "test": "Wilcoxon signed-rank",
                        "statistic": stat,
                        "p_value": p_value,
                        "mean_diff_a_minus_b": float(diffs.mean()),
                        "median_diff_a_minus_b": float(diffs.median()),
                    }
                )
                raw_p_values.append(p_value)
                pairwise_indices.append(len(pairwise_rows))
            else:
                row.update(
                    {
                        "test": "Wilcoxon signed-rank",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "mean_diff_a_minus_b": np.nan,
                        "median_diff_a_minus_b": np.nan,
                    }
                )
            pairwise_rows.append(row)

        adjusted = benjamini_hochberg(raw_p_values)
        for idx, adj_p in zip(pairwise_indices, adjusted):
            pairwise_rows[idx]["p_value_fdr_bh"] = adj_p
        for idx in set(range(len(pairwise_rows))) - set(pairwise_indices):
            if (
                pairwise_rows[idx]["brain_region"] == brain_region
                and pairwise_rows[idx]["syllable"] == int(syllable)
                and "p_value_fdr_bh" not in pairwise_rows[idx]
            ):
                pairwise_rows[idx]["p_value_fdr_bh"] = np.nan

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def run_social_context_tests(subject_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (brain_region, intruder_identity, syllable), group in subject_level.groupby(
        ["brain_region", "intruder_identity", "syllable"], observed=True
    ):
        wide = group.pivot(index="subject_name", columns="social_context", values="mean_DA")
        row = {
            "brain_region": brain_region,
            "intruder_identity": intruder_identity,
            "syllable": int(syllable),
            "n_subjects_paired": int(wide.dropna().shape[0]),
        }

        if {"social", "nonsocial"}.issubset(wide.columns):
            paired = wide[["social", "nonsocial"]].dropna()
        else:
            paired = pd.DataFrame(columns=["social", "nonsocial"])

        if paired.shape[0] >= MIN_PAIRED_SUBJECTS:
            stat, p_value = safe_wilcoxon(paired["social"], paired["nonsocial"])
            diffs = paired["social"] - paired["nonsocial"]
            row.update(
                {
                    "test": "Wilcoxon signed-rank",
                    "statistic": stat,
                    "p_value": p_value,
                    "mean_diff_social_minus_nonsocial": float(diffs.mean()),
                    "median_diff_social_minus_nonsocial": float(diffs.median()),
                    "mean_social": float(paired["social"].mean()),
                    "mean_nonsocial": float(paired["nonsocial"].mean()),
                }
            )
        else:
            row.update(
                {
                    "test": "Wilcoxon signed-rank",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "mean_diff_social_minus_nonsocial": np.nan,
                    "median_diff_social_minus_nonsocial": np.nan,
                    "mean_social": np.nan,
                    "mean_nonsocial": np.nan,
                }
            )
        rows.append(row)

    results = pd.DataFrame(rows)

    adjusted_blocks: list[pd.DataFrame] = []
    for (brain_region, intruder_identity), block in results.groupby(
        ["brain_region", "intruder_identity"], sort=False, observed=True
    ):
        block = block.copy()
        valid_mask = block["p_value"].notna()
        block.loc[valid_mask, "p_value_fdr_bh"] = benjamini_hochberg(block.loc[valid_mask, "p_value"].tolist())
        block.loc[~valid_mask, "p_value_fdr_bh"] = np.nan
        adjusted_blocks.append(block)

    return pd.concat(adjusted_blocks, ignore_index=True)


def write_outputs(
    top_syllables: list[int],
    subject_level: pd.DataFrame,
    agent_omnibus: pd.DataFrame,
    agent_pairwise: pd.DataFrame,
    social_stats: pd.DataFrame,
) -> None:
    pd.DataFrame(
        {
            "rank": range(1, len(top_syllables) + 1),
            "syllable": top_syllables,
        }
    ).to_csv(ROOT / "top10_syllables_ranked.csv", index=False)

    subject_level.to_csv(ROOT / "subject_level_mean_da_top10.csv", index=False)
    agent_omnibus.to_csv(ROOT / "stats_top10_syllable_x_agent_omnibus.csv", index=False)
    agent_pairwise.to_csv(ROOT / "stats_top10_syllable_x_agent_pairwise.csv", index=False)
    social_stats.to_csv(ROOT / "stats_top10_social_vs_nonsocial.csv", index=False)

    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
        summary_df["syllable"] = summary_df["syllable"].astype(int)
        summary_df = summary_df[summary_df["syllable"].isin(top_syllables)].copy()
        summary_df.to_csv(ROOT / "syllable_mean_da_by_region_and_social_context_top10.csv", index=False)

    if HEATMAP_CSV.exists():
        heatmap_df = pd.read_csv(HEATMAP_CSV)
        top10_columns = ["brain_region", "social_context"] + [str(s) for s in top_syllables]
        existing_columns = [col for col in top10_columns if col in heatmap_df.columns]
        heatmap_df.loc[:, existing_columns].to_csv(ROOT / "syllable_mean_da_heatmap_matrix_top10.csv", index=False)


def main() -> None:
    df = pd.read_csv(
        INPUT_CSV,
        usecols=[
            "brain_region",
            "subject_name",
            "intruder_identity",
            "social_context",
            "syllable",
            "zscore_DA",
        ],
    )
    df = df.dropna(
        subset=["brain_region", "subject_name", "intruder_identity", "social_context", "syllable", "zscore_DA"]
    ).copy()

    top_syllables = get_top_syllables(df, TOP_N)
    subject_level = build_subject_level_table(df, top_syllables)
    agent_omnibus, agent_pairwise = run_agent_tests(subject_level)
    social_stats = run_social_context_tests(subject_level)

    write_outputs(top_syllables, subject_level, agent_omnibus, agent_pairwise, social_stats)

    print(f"Top {TOP_N} syllables: {top_syllables}")
    print(f"Wrote subject-level table: {ROOT / 'subject_level_mean_da_top10.csv'}")
    print(f"Wrote agent omnibus stats: {ROOT / 'stats_top10_syllable_x_agent_omnibus.csv'}")
    print(f"Wrote agent pairwise stats: {ROOT / 'stats_top10_syllable_x_agent_pairwise.csv'}")
    print(f"Wrote social vs nonsocial stats: {ROOT / 'stats_top10_social_vs_nonsocial.csv'}")


if __name__ == "__main__":
    main()
