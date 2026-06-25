from trial_class import *
import os
import re

import numpy as np
import pandas as pd
import tdt

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from trial_class import *
from bouts_extension import (
    get_trial_dataframes,
    create_first_investigation_bout_summary_df,
    plot_first_investigation_da_vs_total_duration,
)

import matplotlib.pyplot as plt
import re
from scipy.stats import ttest_rel
from scipy.stats import linregress, pearsonr, ttest_ind
import matplotlib.cm as cm



def trim_short_term_to_5min(trial_data, short_term_bout='Short_Term-1', max_duration=300):
    """
    Trims the 'Short_Term-1' bout to only include behavior events within the first 5 minutes (300 seconds)
    for each subject. Returns a modified trial_data dictionary compatible with create_metadata_dataframe.

    Parameters
    ----------
    trial_data : dict
        Dictionary of {subject_id : DataFrame}, from get_trial_dataframes().
    
    short_term_bout : str
        The name of the bout to trim (default is 'Short_Term-1').
    
    max_duration : int or float
        The maximum duration in seconds to retain (default 300 seconds = 5 minutes).

    Returns
    -------
    trimmed_data : dict
        Updated trial_data dictionary with trimmed 'Short_Term-1' bout for each subject.
    """
    trimmed_data = {}

    for subject_id, df in trial_data.items():
        df_copy = df.copy()

        # Filter only Short_Term-1 rows
        st_mask = df_copy["Bout"] == short_term_bout
        df_st = df_copy[st_mask]

        if not df_st.empty:
            # Find the starting time for Short_Term-1
            start_time = df_st["Event_Start"].min()
            cutoff_time = start_time + max_duration  # 5 minutes after first event start

            # Trim to only events within the first 5 minutes
            df_st_trimmed = df_st[df_st["Event_Start"] <= cutoff_time].copy()

            # Combine with the rest of the DataFrame (non-Short_Term-1 rows)
            df_other = df_copy[~st_mask]
            df_combined = pd.concat([df_other, df_st_trimmed], ignore_index=True)
        else:
            # If no Short_Term-1 events, retain original DataFrame
            df_combined = df_copy

        trimmed_data[subject_id] = df_combined

    return trimmed_data 


def _normalize_mouse_id(mouse_id):
    if pd.isna(mouse_id):
        return None
    return str(mouse_id).strip().lower()


def _load_rank_dataframe(rank_source):
    """
    Accept either a DataFrame with rank columns or a CSV/XLSX path and return a
    normalized DataFrame with columns ['Subject', 'Rank'].
    """
    if isinstance(rank_source, pd.DataFrame):
        rank_df = rank_source.copy()
    elif isinstance(rank_source, str):
        if rank_source.lower().endswith((".xlsx", ".xls")):
            rank_df = pd.read_excel(rank_source)
        else:
            rank_df = pd.read_csv(rank_source)
    else:
        raise ValueError("rank_source must be a pandas DataFrame or a path to a CSV/XLSX file.")

    rename_map = {}
    for col in rank_df.columns:
        col_norm = str(col).strip().lower()
        if col_norm in {"subject", "id", "mouse_identity"}:
            rename_map[col] = "Subject"
        elif col_norm == "rank":
            rename_map[col] = "Rank"

    rank_df = rank_df.rename(columns=rename_map)
    if "Subject" not in rank_df.columns or "Rank" not in rank_df.columns:
        raise ValueError("rank_source must contain subject/id and rank columns.")

    rank_df = rank_df[["Subject", "Rank"]].copy()
    rank_df["Subject"] = rank_df["Subject"].map(_normalize_mouse_id)
    rank_df = rank_df.dropna(subset=["Subject", "Rank"]).drop_duplicates(subset=["Subject"], keep="last")
    return rank_df


def load_long_term_cagemate_mapping(mapping_source):
    """
    Load the home-cage long-term cagemate workbook and return a subject->agent dict.
    """
    if isinstance(mapping_source, pd.DataFrame):
        mapping_df = mapping_source.copy()
    else:
        mapping_df = pd.read_excel(mapping_source)

    rename_map = {}
    for col in mapping_df.columns:
        col_norm = str(col).strip().lower()
        if col_norm == "subject":
            rename_map[col] = "Subject"
        elif col_norm == "cagemate":
            rename_map[col] = "Cagemate"

    mapping_df = mapping_df.rename(columns=rename_map)
    if "Subject" not in mapping_df.columns or "Cagemate" not in mapping_df.columns:
        raise ValueError("Mapping file must contain 'Subject' and 'Cagemate' columns.")

    mapping_df["Subject"] = mapping_df["Subject"].map(_normalize_mouse_id)
    mapping_df["Cagemate"] = mapping_df["Cagemate"].map(_normalize_mouse_id)
    mapping_df = mapping_df.dropna(subset=["Subject", "Cagemate"])
    return dict(zip(mapping_df["Subject"], mapping_df["Cagemate"]))


def find_ranks_using_ds(
    file_path,
    subject_col="subject",
    agent_col="agent",
    subject_wins_col="mouse_1_wins",
    agent_wins_col="mouse_2_wins",
    drop_ids=None,
):
    """
    Standalone David's-score rank calculator for home-cage or other pairwise
    interaction tables.

    Expected columns by default:
    - subject
    - agent
    - mouse_1_wins
    - mouse_2_wins

    Returns a DataFrame with columns:
    ['ID', 'DS', 'Cage', 'Rank']
    """
    if drop_ids is None:
        drop_ids = ["n8"]

    def _creating_new_df(individuals_array):
        columns = ["ID"]
        w_array = [f"{mouse_id}w" for mouse_id in individuals_array]
        m_array = [f"{mouse_id}m" for mouse_id in individuals_array]
        columns.extend(w_array)
        columns.extend(m_array)
        calculations = ["w", "l", "w2", "l2", "DS"]
        columns.extend(calculations)

        empty_df = pd.DataFrame(columns=columns)
        empty_df["ID"] = individuals_array
        new_df = empty_df.fillna(0).infer_objects(copy=False)
        new_df[calculations] = new_df[calculations].astype(float)
        return new_df, w_array, m_array

    df = pd.read_excel(file_path, header=0)
    df.columns = [str(col).lower().strip().replace(" ", "_") for col in df.columns]

    subject_col = subject_col.lower().strip().replace(" ", "_")
    agent_col = agent_col.lower().strip().replace(" ", "_")
    subject_wins_col = subject_wins_col.lower().strip().replace(" ", "_")
    agent_wins_col = agent_wins_col.lower().strip().replace(" ", "_")

    required_cols = [subject_col, agent_col, subject_wins_col, agent_wins_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for David's score calculation: {missing_cols}")

    df = df[required_cols].copy()
    df[subject_col] = df[subject_col].map(_normalize_mouse_id)
    df[agent_col] = df[agent_col].map(_normalize_mouse_id)
    df = df.dropna(subset=[subject_col, agent_col])

    individuals = sorted(set(df[subject_col]).union(set(df[agent_col])))
    individuals_array = np.array(individuals)
    new_df, w_array, m_array = _creating_new_df(individuals_array)
    id_to_idx = {mouse_id: idx for idx, mouse_id in enumerate(individuals)}

    for _, row in df.iterrows():
        mouse1 = row[subject_col]
        mouse2 = row[agent_col]
        idx1 = id_to_idx[mouse1]
        idx2 = id_to_idx[mouse2]

        mouse1_wins = int(row[subject_wins_col]) if not pd.isna(row[subject_wins_col]) else 0
        mouse2_wins = int(row[agent_wins_col]) if not pd.isna(row[agent_wins_col]) else 0
        total_matches = mouse1_wins + mouse2_wins

        new_df.loc[idx1, f"{mouse2}w"] += mouse1_wins
        new_df.loc[idx2, f"{mouse1}w"] += mouse2_wins
        new_df.loc[idx1, f"{mouse2}m"] += total_matches
        new_df.loc[idx2, f"{mouse1}m"] += total_matches

    for i in range(len(individuals_array)):
        w_val = 0.0
        l_val = 0.0
        for j in range(len(individuals_array)):
            num_wins = new_df.at[i, w_array[j]]
            num_matches = new_df.at[i, m_array[j]]
            if pd.isna(num_matches) or num_matches == 0:
                continue
            if pd.isna(num_wins):
                num_wins = 0
            p_win = num_wins / num_matches
            p_loss = 1 - p_win
            w_val += p_win
            l_val += p_loss
        new_df.loc[i, ["w", "l"]] = float(w_val), float(l_val)

    for i in range(len(individuals_array)):
        w2_val = 0.0
        l2_val = 0.0
        for j in range(len(individuals_array)):
            num_wins = new_df.at[i, w_array[j]]
            num_matches = new_df.at[i, m_array[j]]
            if pd.isna(num_matches) or num_matches == 0:
                continue
            if pd.isna(num_wins):
                num_wins = 0
            p_win = num_wins / num_matches
            p_loss = 1 - p_win
            w_opp = new_df.at[j, "w"]
            l_opp = new_df.at[j, "l"]
            w2_val += p_win * w_opp
            l2_val += p_loss * l_opp
        new_df.at[i, "w2"] = float(w2_val)
        new_df.at[i, "l2"] = float(l2_val)

    new_df["DS"] = new_df["w"] + new_df["w2"] - new_df["l"] - new_df["l2"]
    new_df = new_df[["ID", "DS"]].copy()

    if drop_ids:
        drop_ids_norm = {_normalize_mouse_id(mouse_id) for mouse_id in drop_ids}
        new_df = new_df.loc[~new_df["ID"].isin(drop_ids_norm)].copy()

    new_df["Prefix"] = new_df["ID"].str.extract(r"([a-zA-Z]+)")
    new_df["Number"] = new_df["ID"].str.extract(r"(\d+)").astype(int)
    new_df["Cage"] = new_df.apply(lambda row: f"{row['Prefix']}{1 if row['Number'] <= 4 else 2}", axis=1)
    new_df["Rank"] = new_df.groupby("Cage")["DS"].rank(ascending=False, method="min").astype("Int64")
    new_df = new_df.drop(columns=["Prefix", "Number"])
    return new_df


def find_home_cage_ranks_using_ds(file_path, **kwargs):
    """
    Convenience alias for David's-score ranking in home-cage analyses.
    """
    return find_ranks_using_ds(file_path, **kwargs)


def build_home_cage_long_term_rank_summary(
    experiment,
    rank_source,
    mapping_source,
    behavior="Investigation",
    bout_name="Long_Term-1",
    da_col="Mean Z-score",
):
    """
    Build one row per subject for the long-term home-cage bout, annotating the
    mapped agent's absolute rank and whether that agent is higher or lower rank
    than the resident subject.
    """
    if da_col not in {"Mean Z-score", "AUC", "Max Peak"}:
        raise ValueError("da_col must be one of 'Mean Z-score', 'AUC', or 'Max Peak'.")

    rank_df = _load_rank_dataframe(rank_source)
    rank_dict = dict(zip(rank_df["Subject"], rank_df["Rank"]))
    cagemate_dict = load_long_term_cagemate_mapping(mapping_source)

    trial_data = get_trial_dataframes(experiment)
    summary_df = create_first_investigation_bout_summary_df(
        trial_data=trial_data,
        behavior=behavior,
        desired_bouts=[bout_name],
        group_label_map={bout_name: bout_name},
        group_col="BoutGroup",
    )

    if summary_df.empty:
        return summary_df

    summary_df = summary_df.copy()
    summary_df["Subject"] = summary_df["Subject"].map(_normalize_mouse_id)
    summary_df["Subject Rank"] = summary_df["Subject"].map(rank_dict)
    summary_df["Agent"] = summary_df["Subject"].map(cagemate_dict)
    summary_df["Agent Rank"] = summary_df["Agent"].map(rank_dict)

    def classify_relative_rank(row):
        subj_rank = row["Subject Rank"]
        agent_rank = row["Agent Rank"]
        if pd.isna(subj_rank) or pd.isna(agent_rank):
            return np.nan
        if agent_rank < subj_rank:
            return "Higher-ranked agent"
        if agent_rank > subj_rank:
            return "Lower-ranked agent"
        return "Equal-ranked agent"

    summary_df["Relative Rank"] = summary_df.apply(classify_relative_rank, axis=1)
    summary_df["Rank Difference"] = summary_df["Agent Rank"] - summary_df["Subject Rank"]
    summary_df["DA"] = summary_df[da_col]
    summary_df["Bout"] = bout_name

    cols = [
        "Subject",
        "Bout",
        "Agent",
        "Subject Rank",
        "Agent Rank",
        "Relative Rank",
        "Rank Difference",
        "First Investigation Duration",
        "Total Investigation Duration",
        "DA",
        "AUC",
        "Max Peak",
        "Mean Z-score",
    ]
    return summary_df[cols].copy()


def plot_home_cage_long_term_da_by_relative_rank(
    rank_summary_df,
    da_col="DA",
    group_col="Relative Rank",
    group_order=None,
    group_colors=None,
    title="Home Cage Long-Term DA by Relative Rank",
    ylabel="Mean Z-scored ΔF/F during 1st investigation",
    figsize=(8, 7),
    ax=None,
):
    """
    Plot long-term first-investigation DA grouped by whether the mapped agent is
    higher or lower rank than the subject.
    """
    plot_df = rank_summary_df.dropna(subset=[group_col, da_col]).copy()
    if plot_df.empty:
        raise ValueError("No valid rows remain for relative-rank plotting.")

    if group_order is None:
        group_order = ["Higher-ranked agent", "Lower-ranked agent", "Equal-ranked agent"]
    group_order = [g for g in group_order if g in plot_df[group_col].unique()]

    if group_colors is None:
        group_colors = {
            "Higher-ranked agent": "#c44e52",
            "Lower-ranked agent": "#4c72b0",
            "Equal-ranked agent": "#7f7f7f",
        }

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    means = []
    sems = []
    x_positions = np.arange(len(group_order))

    for idx, group_name in enumerate(group_order):
        vals = plot_df.loc[plot_df[group_col] == group_name, da_col].dropna().to_numpy(dtype=float)
        means.append(np.nanmean(vals))
        sems.append(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0] * len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=120,
            color=group_colors.get(group_name, "gray"),
            edgecolor="black",
            linewidth=1.5,
            alpha=0.95,
            zorder=3,
        )

    ax.bar(
        x_positions,
        means,
        yerr=sems,
        capsize=6,
        width=0.62,
        color=[group_colors.get(g, "gray") for g in group_order],
        edgecolor="black",
        linewidth=2.5,
        alpha=0.35,
        zorder=2,
    )

    if len(group_order) >= 2:
        first_vals = plot_df.loc[plot_df[group_col] == group_order[0], da_col].dropna().to_numpy(dtype=float)
        second_vals = plot_df.loc[plot_df[group_col] == group_order[1], da_col].dropna().to_numpy(dtype=float)
        if len(first_vals) > 1 and len(second_vals) > 1:
            _, p_val = ttest_ind(first_vals, second_vals, equal_var=False, nan_policy="omit")
            ax.text(0.02, 0.96, f"p = {p_val:.3g}", transform=ax.transAxes, va="top", fontsize=15)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_order, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis="y", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)

    if created_fig:
        plt.tight_layout()

    return plot_df, fig, ax


def plot_home_cage_long_term_da_vs_rank(
    rank_summary_df,
    rank_col="Agent Rank",
    da_col="DA",
    title=None,
    xlabel=None,
    ylabel="Mean Z-scored ΔF/F during 1st investigation",
    figsize=(8, 7),
    ax=None,
):
    """
    Scatter plot and correlation line for long-term home-cage DA vs rank.
    Use rank_col='Agent Rank' for agent effects or rank_col='Subject Rank' for
    resident-subject effects.
    """
    plot_df = rank_summary_df.dropna(subset=[rank_col, da_col]).copy()
    if plot_df.empty:
        raise ValueError("No valid rows remain for rank-correlation plotting.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.scatter(
        plot_df[rank_col],
        plot_df[da_col],
        s=150,
        facecolors="none",
        edgecolors="#2f2f2f",
        linewidth=2.2,
        zorder=3,
    )

    r_text = "r = ---"
    p_text = "p = ---"
    if len(plot_df) > 1 and plot_df[rank_col].nunique() > 1:
        x_vals = plot_df[rank_col].to_numpy(dtype=float)
        y_vals = plot_df[da_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_fit = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 100)
        ax.plot(x_fit, slope * x_fit + intercept, color="black", linewidth=2.5, linestyle=(0, (8, 6)), zorder=2)
        r_val, p_val = pearsonr(x_vals, y_vals)
        r_text = f"r = {r_val:.3f}"
        p_text = f"p = {p_val:.3g}"

    ax.text(0.04, 0.96, f"{r_text}\n{p_text}\nn = {len(plot_df)} mice", transform=ax.transAxes, va="top", fontsize=15)
    ax.set_xlabel(xlabel if xlabel else rank_col, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if title is None:
        title = f"Home Cage Long-Term DA vs {rank_col}"
    ax.set_title(title, fontsize=20)
    ax.set_xticks(sorted(plot_df[rank_col].dropna().unique()))
    ax.tick_params(axis="both", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)

    if created_fig:
        plt.tight_layout()

    return plot_df, fig, ax


def plot_home_cage_long_term_da_vs_rank_colored(
    rank_summary_df,
    rank_col="Agent Rank",
    da_col="DA",
    color_col="Relative Rank",
    color_order=None,
    color_map=None,
    legend_title=None,
    title=None,
    xlabel=None,
    ylabel="Mean Z-scored ΔF/F during 1st investigation",
    figsize=(8, 7),
    ax=None,
):
    """
    Scatter plot and correlation line for long-term home-cage DA vs rank, with
    points color-coded by a grouping column such as 'Relative Rank'.
    """
    subset_cols = [rank_col, da_col, color_col]
    plot_df = rank_summary_df.dropna(subset=subset_cols).copy()
    if plot_df.empty:
        raise ValueError("No valid rows remain for colored rank-correlation plotting.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    if color_map is None:
        color_map = {
            "Higher-ranked agent": "#c44e52",
            "Lower-ranked agent": "#4c72b0",
            "Equal-ranked agent": "#7f7f7f",
        }
    if color_order is None:
        color_order = list(dict.fromkeys(plot_df[color_col].tolist()))

    for color_name in color_order:
        group_df = plot_df[plot_df[color_col] == color_name]
        if group_df.empty:
            continue
        ax.scatter(
            group_df[rank_col],
            group_df[da_col],
            s=150,
            facecolors="none",
            edgecolors=color_map.get(color_name, "#2f2f2f"),
            linewidth=2.2,
            zorder=3,
            label=str(color_name),
        )

    r_text = "r = ---"
    p_text = "p = ---"
    if len(plot_df) > 1 and plot_df[rank_col].nunique() > 1:
        x_vals = plot_df[rank_col].to_numpy(dtype=float)
        y_vals = plot_df[da_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_fit = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 100)
        ax.plot(
            x_fit,
            slope * x_fit + intercept,
            color="black",
            linewidth=2.5,
            linestyle=(0, (8, 6)),
            zorder=2,
        )
        r_val, p_val = pearsonr(x_vals, y_vals)
        r_text = f"r = {r_val:.3f}"
        p_text = f"p = {p_val:.3g}"

    ax.text(0.04, 0.96, f"{r_text}\n{p_text}\nn = {len(plot_df)} mice", transform=ax.transAxes, va="top", fontsize=15)
    ax.set_xlabel(xlabel if xlabel else rank_col, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if title is None:
        title = f"Home Cage Long-Term DA vs {rank_col}"
    ax.set_title(title, fontsize=20)
    ax.set_xticks(sorted(plot_df[rank_col].dropna().unique()))
    ax.tick_params(axis="both", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)

    if legend_title is None:
        legend_title = color_col
    ax.legend(title=legend_title, fontsize=12, title_fontsize=13, frameon=False)

    if created_fig:
        plt.tight_layout()

    return plot_df, fig, ax


def plot_home_cage_long_term_da_vs_rank_difference(
    rank_summary_df,
    diff_col="Rank Difference",
    da_col="DA",
    title="Home Cage Long-Term DA by Relative Rank Difference",
    xlabel="Agent Rank - Subject Rank",
    ylabel="Mean Z-scored Î”F/F during 1st investigation",
    figsize=(8, 7),
    ax=None,
):
    """
    Scatter plot of first-investigation DA against the numeric rank difference
    between the agent and the subject.
    """
    plot_df = rank_summary_df.dropna(subset=[diff_col, da_col]).copy()
    if plot_df.empty:
        raise ValueError("No valid rows remain for rank-difference plotting.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.scatter(
        plot_df[diff_col],
        plot_df[da_col],
        s=150,
        facecolors="none",
        edgecolors="#2f2f2f",
        linewidth=2.2,
        zorder=3,
    )

    r_text = "r = ---"
    p_text = "p = ---"
    if len(plot_df) > 1 and plot_df[diff_col].nunique() > 1:
        x_vals = plot_df[diff_col].to_numpy(dtype=float)
        y_vals = plot_df[da_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_fit = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 100)
        ax.plot(
            x_fit,
            slope * x_fit + intercept,
            color="black",
            linewidth=2.5,
            linestyle=(0, (8, 6)),
            zorder=2,
        )
        r_val, p_val = pearsonr(x_vals, y_vals)
        r_text = f"r = {r_val:.3f}"
        p_text = f"p = {p_val:.3g}"

    ax.axvline(0, color="#9a9a9a", linewidth=1.8, linestyle="--", zorder=1)
    ax.text(0.04, 0.96, f"{r_text}\n{p_text}\nn = {len(plot_df)} mice", transform=ax.transAxes, va="top", fontsize=15)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(sorted(plot_df[diff_col].dropna().unique()))
    ax.tick_params(axis="both", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)

    if created_fig:
        plt.tight_layout()

    return plot_df, fig, ax


def plot_home_cage_long_term_da_rank_interaction(
    rank_summary_df,
    da_col="DA",
    subject_rank_col="Subject Rank",
    agent_rank_col="Agent Rank",
    title="Home Cage Long-Term DA by Subject Rank x Agent Rank",
    xlabel="Agent Rank",
    ylabel="Mean Z-scored Î”F/F during 1st investigation",
    figsize=(10, 8),
    ax=None,
    cmap_name="viridis",
):
    """
    Interaction-style line plot showing mean DA across agent ranks, with one
    line per subject-rank group.
    """
    plot_df = rank_summary_df.dropna(subset=[subject_rank_col, agent_rank_col, da_col]).copy()
    if plot_df.empty:
        raise ValueError("No valid rows remain for subject-rank x agent-rank interaction plotting.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    grouped = (
        plot_df.groupby([subject_rank_col, agent_rank_col])[da_col]
        .agg(mean="mean", count="count", std="std")
        .reset_index()
    )
    grouped["sem"] = grouped.apply(
        lambda row: row["std"] / np.sqrt(row["count"]) if row["count"] > 1 and pd.notna(row["std"]) else 0,
        axis=1,
    )

    subject_ranks = sorted(plot_df[subject_rank_col].dropna().unique())
    agent_ranks = sorted(plot_df[agent_rank_col].dropna().unique())
    cmap = cm.get_cmap(cmap_name, len(subject_ranks))

    for idx, subj_rank in enumerate(subject_ranks):
        subj_df = grouped[grouped[subject_rank_col] == subj_rank].sort_values(agent_rank_col)
        color = cmap(idx)
        ax.plot(
            subj_df[agent_rank_col],
            subj_df["mean"],
            marker="o",
            markersize=9,
            linewidth=2.5,
            color=color,
            label=f"Subject rank {int(subj_rank)}",
            zorder=3,
        )
        ax.errorbar(
            subj_df[agent_rank_col],
            subj_df["mean"],
            yerr=subj_df["sem"],
            fmt="none",
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            zorder=2,
        )

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(agent_ranks)
    ax.tick_params(axis="both", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.legend(title="Subject rank", fontsize=13, title_fontsize=14, frameon=False)

    if created_fig:
        plt.tight_layout()

    return plot_df, grouped, fig, ax


def plot_home_cage_first_investigation_vs_total_duration(
    experiment,
    cap_acq_st_5min=False,
    behavior="Investigation",
    desired_bouts=None,
    group_col="Agent",
    da_col="Mean Z-score",
    title="",
    xlabel="Mean Z-scored ΔF/F during 1st investigation",
    ylabel="Total Investigation Duration (s)",
    group_colors=None,
    ax=None,
    save=False,
    save_name=None,
    pad_inches=0.1,
    **plot_kwargs,
):
    """
    Plot first-investigation DA vs total investigation duration for home-cage
    bouts, grouped by agent identity.

    If ``cap_acq_st_5min`` is True, only ``Short_Term-1`` is trimmed to the
    first 5 minutes before bout totals are computed.
    """
    if desired_bouts is None:
        desired_bouts = ["Short_Term-1", "Short_Term-2", "Long_Term-1", "Novel-1"]

    if group_colors is None:
        group_colors = {
            "Acq-ST": "#1f77b4",
            "Short Term": "#4f86f7",
            "Long Term": "#b08b3a",
            "Novel": "#d948c5",
        }

    agent_labels = {
        "Short_Term-1": "Acq-ST",
        "Short_Term-2": "Short Term",
        "Long_Term-1": "Long Term",
        "Novel-1": "Novel",
    }

    trial_data = get_trial_dataframes(experiment)
    if cap_acq_st_5min:
        trial_data = trim_short_term_to_5min(trial_data, short_term_bout="Short_Term-1", max_duration=300)

    summary_df = create_first_investigation_bout_summary_df(
        trial_data=trial_data,
        behavior=behavior,
        desired_bouts=desired_bouts,
        group_label_map=agent_labels,
        group_col=group_col,
    )

    plot_df, fig, ax = plot_first_investigation_da_vs_total_duration(
        summary_df=summary_df,
        da_col=da_col,
        duration_col="Total Investigation Duration",
        group_col=group_col,
        group_order=["Acq-ST", "Short Term", "Long Term", "Novel"],
        group_colors=group_colors,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        save=save,
        save_name=save_name,
        pad_inches=pad_inches,
        **plot_kwargs,
    )

    return summary_df, plot_df, fig, ax


def plot_home_cage_first_investigation_vs_total_duration_comparison(
    experiment,
    behavior="Investigation",
    desired_bouts=None,
    da_col="Mean Z-score",
    group_colors=None,
    figsize=(18, 7),
    save=False,
    save_name=None,
    pad_inches=0.1,
    **plot_kwargs,
):
    """
    Plot uncapped vs 5-minute capped Acq-ST home-cage correlations side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    uncapped_summary, uncapped_plot, _, _ = plot_home_cage_first_investigation_vs_total_duration(
        experiment=experiment,
        cap_acq_st_5min=False,
        behavior=behavior,
        desired_bouts=desired_bouts,
        da_col=da_col,
        title="Home Cage: Acq-ST Uncapped",
        group_colors=group_colors,
        ax=axes[0],
        **plot_kwargs,
    )

    capped_summary, capped_plot, _, _ = plot_home_cage_first_investigation_vs_total_duration(
        experiment=experiment,
        cap_acq_st_5min=True,
        behavior=behavior,
        desired_bouts=desired_bouts,
        da_col=da_col,
        title="Home Cage: Acq-ST Capped at 5 min",
        group_colors=group_colors,
        ax=axes[1],
        **plot_kwargs,
    )

    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=pad_inches)

    return {
        "uncapped_summary": uncapped_summary,
        "uncapped_plot": uncapped_plot,
        "capped_summary": capped_summary,
        "capped_plot": capped_plot,
        "fig": fig,
        "axes": axes,
    }




# -------------------------------------------------------------------
# Rank stuff. We'll be working on this more: not in documentation yet.
# Plots DA vs. bout duration plots
def plot_da_vs_duration_by_agent(experiment, 
                                 agents_of_interest, 
                                 agent_colors, 
                                 agent_labels, 
                                 title,
                                 da_metric='Mean Z-score',
                                 figsize=(10, 7),
                                 ylim=None,
                                 yticks_increment=None,
                                 xlabel = None,
                                 legend_loc='upper left',
                                 pad_inches=0.1,
                                 save=None,
                                 save_name=None):  # New parameter
    """
    Plot correlation between event-induced DA (Mean Z-score, AUC, or Max Peak) 
    and bout duration for selected agents.
    """
    valid_metrics = ['Mean Z-score', 'AUC', 'Max Peak']
    if da_metric not in valid_metrics:
        raise ValueError(f"Invalid da_metric. Choose from {valid_metrics}")

    trial_dfs = get_trial_dataframes(experiment)
    points = []

    for trial_id, df in zip(experiment.trials.keys(), trial_dfs):
        for bout_name in agents_of_interest:
            subset = df[(df["Bout"] == bout_name) & (df["Behavior"] == "Investigation")]
            if subset.empty:
                continue

            first_invest = subset.iloc[0]
            duration = first_invest["Duration (s)"]
            mean_z = first_invest.get("Mean Z-score", np.nan)
            auc = first_invest.get("AUC", np.nan)
            max_peak = first_invest.get("Max Peak", np.nan)

            prefix = bout_name.split('-')[0]
            agent_label = agent_labels.get(prefix, prefix)

            points.append({
                'Subject': trial_id,
                'Agent': agent_label,
                'Bout_Duration': duration,
                'Mean Z-score': mean_z,
                'AUC': auc,
                'Max Peak': max_peak
            })

    points_df = pd.DataFrame(points)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    all_x, all_y = [], []

    for agent, group in points_df.groupby("Agent"):
        x = group[da_metric].values
        y = group["Bout_Duration"].values
        all_x.extend(x)
        all_y.extend(y)

        color = agent_colors.get(agent, 'gray')
        ax.scatter(x, y, color=color, s=250, alpha=1.0, edgecolor='black', linewidth=3, label=agent, zorder=3)

    # --- Regression Line (Pooled) ---
    stats_text_lines = ["r = ---", "p = ---", "n = ---"]
    if len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_fit = np.linspace(min(all_x), max(all_x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='black', linewidth=2.5, linestyle='-', zorder=2)

        stats_text_lines = [
            f"r = {r_val:.3f}",
            f"p = {p_val:.3f}",
            f"n = {len(all_x)} events"
        ]

    # --- Labels ---
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel("Bout Duration (s)", fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=24)

    # X-axis label
    final_xlabel = xlabel if xlabel else da_metric
    ax.set_xlabel(final_xlabel, fontsize=24)


    # --- Y-axis formatting ---
    if ylim:
        ax.set_ylim(ylim)
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        yticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if x.is_integer() else f'{x}'))

    # --- Combined Legend ---
    handles, labels = ax.get_legend_handles_labels()
    stats_label = "\n".join(stats_text_lines)
    stats_handle = plt.Line2D([], [], color='none', label=stats_label)
    handles.append(stats_handle)
    labels.append(stats_label)

    legend = ax.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=16, title='Agent', title_fontsize=18, 
                       frameon=True, facecolor='white', edgecolor='lightgray', fancybox=False)
    legend.get_frame().set_alpha(0.8)

    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    return points_df

# Plots event-induced DA plots in colors by subject
def plot_da_vs_duration_by_agent_colored(experiment, 
                                 agents_of_interest, 
                                 agent_labels, 
                                 title,
                                 da_metric='Mean Z-score',
                                 figsize=(10, 7),
                                 ylim=None,
                                 yticks_increment=None,
                                 xlabel=None,
                                 legend_loc='upper left',
                                 pad_inches=0.1,
                                 save=None,
                                 save_name=None):
    """
    Plot correlation between event-induced DA and bout duration for each subject with unique colors.
    """
    valid_metrics = ['Mean Z-score', 'AUC', 'Max Peak']
    if da_metric not in valid_metrics:
        raise ValueError(f"Invalid da_metric. Choose from {valid_metrics}")

    trial_dfs = get_trial_dataframes(experiment)
    points = []

    for trial_id, df in zip(experiment.trials.keys(), trial_dfs):
        for bout_name in agents_of_interest:
            subset = df[(df["Bout"] == bout_name) & (df["Behavior"] == "Investigation")]
            if subset.empty:
                continue

            first_invest = subset.iloc[0]
            duration = first_invest["Duration (s)"]
            mean_z = first_invest.get("Mean Z-score", np.nan)
            auc = first_invest.get("AUC", np.nan)
            max_peak = first_invest.get("Max Peak", np.nan)

            prefix = bout_name.split('-')[0]
            agent_label = agent_labels.get(prefix, prefix)

            points.append({
                'Subject': trial_id,
                'Agent': agent_label,
                'Bout_Duration': duration,
                'Mean Z-score': mean_z,
                'AUC': auc,
                'Max Peak': max_peak
            })

    points_df = pd.DataFrame(points)

    # --- Assign unique colors per Subject ---
    unique_subjects = points_df['Subject'].unique()
    color_map = cm.get_cmap('tab20', len(unique_subjects))  # Can change to 'nipy_spectral' etc.
    subject_color_dict = {subj: color_map(i) for i, subj in enumerate(unique_subjects)}

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    all_x, all_y = [], []

    for _, row in points_df.iterrows():
        x = row[da_metric]
        y = row["Bout_Duration"]
        all_x.append(x)
        all_y.append(y)
        subj = row["Subject"]
        color = subject_color_dict[subj]

        ax.scatter(x, y, color=color, s=250, alpha=1.0, edgecolor='black', linewidth=3, label=subj, zorder=3)

    # --- Regression Line (Pooled) ---
    stats_text_lines = ["r = ---", "p = ---", "n = ---"]
    if len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_fit = np.linspace(min(all_x), max(all_x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='black', linewidth=2.5, linestyle='-', zorder=2)

        stats_text_lines = [
            f"r = {r_val:.3f}",
            f"p = {p_val:.3f}",
            f"n = {len(all_x)} events"
        ]

    # --- Axis Labels ---
    final_xlabel = xlabel if xlabel else da_metric
    ax.set_xlabel(final_xlabel, fontsize=24)
    ax.set_ylabel("Bout Duration (s)", fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=24)

    # --- Y-axis formatting ---
    if ylim:
        ax.set_ylim(ylim)
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        yticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if x.is_integer() else f'{x}'))

    # --- Legend per Subject ---
    subject_handles = [
        plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, 
                   markeredgecolor='black', markerfacecolor=color, linewidth=0, label=str(subj))
        for subj, color in subject_color_dict.items()
    ]

    stats_label = "\n".join(stats_text_lines)
    stats_handle = plt.Line2D([], [], color='none', label=stats_label)

    handles_combined = subject_handles + [stats_handle]
    labels_combined = [h.get_label() for h in subject_handles] + [stats_label]

    legend = ax.legend(handles=handles_combined, labels=labels_combined, loc=legend_loc, fontsize=14, 
                       title='Subject ID', title_fontsize=16, frameon=True, facecolor='white', 
                       edgecolor='lightgray', fancybox=False)
    legend.get_frame().set_alpha(0.8)

    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    return points_df

def assign_subject_ranks_to_experiment(experiment, rank_csv_path):
    """
    Loads subject ranks from CSV and assigns each trial’s .behaviors DataFrame a new 'Rank' column.

    Parameters:
    - experiment : Experiment object
    - rank_csv_path : path to CSV with columns ['Subject', 'Rank']
    """
    # Load ranks
    rank_df = _load_rank_dataframe(rank_csv_path)
    rank_dict = dict(zip(rank_df['Subject'].str.lower(), rank_df['Rank']))

    subjects_assigned = 0
    subjects_missing = []

    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            subject_prefix = trial_name[:3].lower()  # Match e.g., 'pp1', 'nn2'

            if subject_prefix in rank_dict:
                trial.behaviors["Rank"] = rank_dict[subject_prefix]
                subjects_assigned += 1
            else:
                trial.behaviors["Rank"] = None
                subjects_missing.append(trial_name)

    print(f"Ranks assigned to {subjects_assigned} trials.")
    if subjects_missing:
        print(f"No rank found for trials: {subjects_missing}")

def generate_investigation_per_agent_df(experiment, rank_csv_path=None, behavior_name='Investigation'):
    """
    Generates a DataFrame with Subject, Rank, and both Total Investigation Time
    and Average Bout Duration per Agent (Bout).

    Returns:
        DataFrame where rows = subjects, columns = [Total_Acq-ST, Avg_Acq-ST, ...], NaNs filled with 0.
    """

    # Load rank CSV if provided
    rank_dict = {}
    if rank_csv_path:
        rank_df = pd.read_csv(rank_csv_path)
        if 'Subject' not in rank_df.columns or 'Rank' not in rank_df.columns:
            rank_df = pd.read_csv(rank_csv_path, header=None, names=['Subject', 'Rank'])
        rank_dict = dict(zip(rank_df['Subject'].str.lower(), rank_df['Rank']))

    data = []
    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            df = trial.behaviors.copy()

            if behavior_name not in df['Behavior'].unique():
                continue

            df = df[df['Behavior'] == behavior_name].copy()
            if df.empty or 'Bout' not in df.columns:
                continue

            # Total duration per Agent
            total_per_agent = df.groupby('Bout')['Duration (s)'].sum()
            # Average bout duration per Agent
            avg_per_agent = df.groupby('Bout')['Duration (s)'].mean()

            # Subject ID
            subj_id = trial_name[:3].lower()
            rank = rank_dict.get(subj_id, None)

            row = {'Subject': subj_id, 'Rank': rank}
            for bout, total_val in total_per_agent.items():
                row[f'Total_{bout}'] = total_val
            for bout, avg_val in avg_per_agent.items():
                row[f'Avg_{bout}'] = avg_val

            data.append(row)

    # Final DataFrame with NaNs filled with 0
    agent_df = pd.DataFrame(data).set_index('Subject').fillna(0)
    return agent_df

def plot_y_across_bouts_ranks(df,  
                             title='Mean Across Bouts', 
                             ylabel='Mean Value', 
                             custom_xtick_labels=None, 
                             custom_xtick_colors=None, 
                             ylim=None, 
                             bar_fill_color='white',     # NEW
                             bar_edge_color='black',     # NEW
                             bar_linewidth=3,            # NEW
                             bar_hatch='///',            # NEW
                             yticks_increment=None, 
                             xlabel='Agent',
                             figsize=(12,7), 
                             pad_inches=0.1,
                             rank_filter=None,
                             metric='Total'):
    """
    Plots mean Total Investigation Time or Average Bout Duration per Agent, with SEM and individual lines.

    Parameters:
        - df (DataFrame): Includes columns like 'Total_X' or 'Avg_X' per agent, plus 'Rank'.
        - metric (str): 'Total' or 'Avg' - determines which columns to plot.
        - rank_filter (int or None): Plot only subjects with this rank (if provided).
        - bar_fill_color (str): Fill color of the bars.
        - bar_edge_color (str): Edge color of the bars.
        - bar_linewidth (float): Width of the bar edges.
        - bar_hatch (str): Hatch pattern for the bars.
        [other params same as before]
    """

    # --- Filter by Rank ---
    if rank_filter is not None:
        if "Rank" not in df.columns:
            print("Rank filtering requested, but 'Rank' column not found.")
            return
        df = df[df["Rank"] == rank_filter]
        if df.empty:
            print(f"No data for Rank {rank_filter}.")
            return
        print(f"Plotting Rank {rank_filter} subjects: {len(df)} entries.")

    # --- Select columns by metric ---
    if metric not in ['Total', 'Avg']:
        raise ValueError("metric must be 'Total' or 'Avg'")
    value_columns = [col for col in df.columns if col.startswith(f"{metric}_")]

    if not value_columns:
        print(f"No columns found starting with '{metric}_'.")
        return

    df_plot = df[value_columns].copy()
    df_plot.columns = [col.replace(f"{metric}_", "") for col in df_plot.columns]

    # --- T-tests ---
    def perform_t_tests(df_vals):
        comparisons = {
            "acq_st_vs_short_term": ("Acq-ST", "Short Term"),
            "acq_st_vs_long_term": ("Acq-ST", "Long Term"),
            "acq_st_vs_novel": ("Acq-ST", "Novel")
        }
        results = {}
        for key, (b1, b2) in comparisons.items():
            if b1 in df_vals.columns and b2 in df_vals.columns:
                paired = df_vals[[b1, b2]].dropna()
                if len(paired) > 1:
                    t_stat, p_value = ttest_rel(paired[b1], paired[b2])
                    results[key] = {"t_stat": t_stat, "p_value": p_value}
        return results

    t_test_results = perform_t_tests(df_plot)

    # --- Stats ---
    mean_vals = df_plot.mean()
    sem_vals = df_plot.sem()

    fig, ax = plt.subplots(figsize=figsize)

    # --- Bar Plot ---
    ax.bar(df_plot.columns, mean_vals, yerr=sem_vals, capsize=6,
           color=bar_fill_color, edgecolor=bar_edge_color, linewidth=bar_linewidth,
           width=0.6, hatch=bar_hatch,
           error_kw=dict(elinewidth=2, capthick=2, zorder=5))

     # --- Lines + Colored Markers ---
    for subject_id, row in df_plot.iterrows():
        subject_prefix = str(subject_id).lower().strip()
        if subject_prefix.startswith('n'):
            marker_color = '#15616F'  # NAc
        elif subject_prefix.startswith('p'):
            marker_color = '#FFAF00'  # mPFC
        else:
            marker_color = 'gray'  # fallback

        # Gray line
        ax.plot(df_plot.columns, row.values, linestyle='-', color='gray',
                alpha=0.5, linewidth=2.5, zorder=1)

        # Colored opaque filled circles, no border, behind error bars
        ax.scatter(df_plot.columns, row.values, color=marker_color,
                   s=120, alpha=1.0, zorder=1)

    # --- Labels ---
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # --- X-ticks ---
    ax.set_xticks(np.arange(len(df_plot.columns)))
    if custom_xtick_labels:
        ax.set_xticklabels(custom_xtick_labels, fontsize=28)
        if custom_xtick_colors:
            for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick.set_color(color)
    else:
        ax.set_xticklabels(df_plot.columns, fontsize=26)

    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)

    # --- Y-limits ---
    all_vals = np.concatenate([df_plot.values.flatten(), mean_vals.values])
    if ylim is None:
        min_val = np.nanmin(all_vals)
        max_val = np.nanmax(all_vals)
        lower_ylim = 0 if min_val > 0 else min_val * 1.1
        upper_ylim = max_val * 1.1
        ax.set_ylim(lower_ylim, upper_ylim)
        if lower_ylim < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)
    else:
        ax.set_ylim(ylim)
        if ylim[0] < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)

    # --- Y-ticks ---
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # --- Aesthetic ---
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # --- Significance Markers ---
    if t_test_results:
        max_y = ax.get_ylim()[1]
        sig_y_offset = max_y * 0.05
        comparisons = {
            "acq_st_vs_short_term": (0, 1),
            "acq_st_vs_long_term": (0, 2),
            "acq_st_vs_novel": (0, 3)
        }
        line_spacing = sig_y_offset * 2.5
        current_y = mean_vals.max() + sig_y_offset

        for key, (x1, x2) in comparisons.items():
            if key in t_test_results:
                p_value = t_test_results[key]["p_value"]
                if p_value < 0.05:
                    significance = "**" if p_value < 0.01 else "*"
                    ax.plot([x1, x2], [current_y, current_y], color='black', linewidth=5)
                    ax.text((x1 + x2) / 2, current_y + sig_y_offset / 1.5, significance,
                            fontsize=40, ha='center', color='black')
                    current_y += line_spacing
                    
    # --- Legend for Region Colors with Dots ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='NAc',
            markerfacecolor='#15616F', markersize=12, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='none', label='mPFC',
            markerfacecolor='#FFAF00', markersize=12, markeredgewidth=0)
    ]

    ax.legend(handles=legend_elements, title="Region", fontsize=20, title_fontsize=22,
            loc='upper right', frameon=True)

    plt.savefig(f'{title}{ylabel[0]}.png', transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    plt.tight_layout(pad=pad_inches)
    plt.show()

def assign_ranks_and_combine_da_metrics(experiment, rank_csv_path):
    """
    Assigns subject ranks (from CSV) to each trial's behaviors DataFrame after DA metrics are computed.
    Then combines all trials into a single DataFrame with columns:
    ['Subject', 'Rank', 'Behavior', 'Bout', 'Event_Start', 'Event_End', 'Duration (s)',
     'AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score', 'Original End', 'Adjusted End']
    
    Returns:
        combined_df (DataFrame): All behaviors + DA metrics + Rank, with Subject column.
    """

    # Load rank CSV or DataFrame
    rank_df = _load_rank_dataframe(rank_csv_path)
    rank_dict = dict(zip(rank_df['Subject'].str.lower(), rank_df['Rank']))

    combined_rows = []
    assigned = 0

    for trial_name, trial in experiment.trials.items():
        subj_id = trial_name[:3].lower()  # e.g., pp1, nn2
        rank = rank_dict.get(subj_id, None)

        if rank is not None and hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            df = trial.behaviors.copy()
            df['Subject'] = subj_id
            df['Rank'] = rank
            combined_rows.append(df)
            assigned += 1
        else:
            print(f"Skipped trial '{trial_name}' — no rank match or empty behaviors.")

    combined_df = pd.concat(combined_rows, ignore_index=True)
    print(f"Ranks assigned to {assigned} trials. Combined DataFrame shape: {combined_df.shape}")
    
    return combined_df

def plot_da_vs_duration_by_agent_flipped(experiment, 
                                         agents_of_interest, 
                                         agent_colors, 
                                         agent_labels, 
                                         title,
                                         da_metric='Mean Z-score',
                                         figsize=(10, 7),
                                         ylim=None,
                                         yticks_increment=None,
                                         ylabel=None,  # Custom Y-axis label
                                         legend_loc='upper left',
                                         pad_inches=0.1,
                                         save=False,
                                         save_name=None):
    """
    Plot correlation between bout duration (X) and event-induced DA (Y) 
    for selected agents, with customizable Y-axis label.

    Also prints skipped Subject × Bout types that had no valid data.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy.stats import linregress
    import pandas as pd

    valid_metrics = ['Mean Z-score', 'AUC', 'Max Peak']
    if da_metric not in valid_metrics:
        raise ValueError(f"Invalid da_metric. Choose from {valid_metrics}")

    trial_data = get_trial_dataframes(experiment)  # List of (trial_id, df)
    points = []
    skipped_points = []

    for trial_id, df in trial_data:
        for bout_name in agents_of_interest:
            subset = df[(df["Bout"] == bout_name) & (df["Behavior"] == "Investigation")]
            if subset.empty:
                skipped_points.append((trial_id, bout_name))
                continue

            first_invest = subset.iloc[0]
            duration = first_invest["Duration (s)"]
            mean_z = first_invest.get("Mean Z-score", np.nan)
            auc = first_invest.get("AUC", np.nan)
            max_peak = first_invest.get("Max Peak", np.nan)

            prefix = bout_name.split('-')[0]
            agent_label = agent_labels.get(prefix, prefix)

            points.append({
                'Subject': trial_id,
                'Agent': agent_label,
                'Bout_Duration': duration,
                'Mean Z-score': mean_z,
                'AUC': auc,
                'Max Peak': max_peak
            })

    points_df = pd.DataFrame(points)

    # --- Print Skipped Data Points ---
    if skipped_points:
        print("⚠️ Skipped Data Points (No valid Investigation rows found):")
        for subj_id, bout in skipped_points:
            print(f"  - Subject: {subj_id}, Bout: {bout}")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    all_x, all_y = [], []

    for agent, group in points_df.groupby("Agent"):
        x = group["Bout_Duration"].values
        y = group[da_metric].values
        all_x.extend(x)
        all_y.extend(y)

        color = agent_colors.get(agent, 'gray')
        ax.scatter(x, y, color=color, s=250, alpha=1.0, edgecolor='black', linewidth=3, label=agent, zorder=3)

    # --- Regression Line ---
    stats_text_lines = ["r = ---", "p = ---", "n = ---"]
    if len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_fit = np.linspace(min(all_x), max(all_x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='black', linewidth=2.5, linestyle='-', zorder=2)

        stats_text_lines = [
            f"r = {r_val:.3f}",
            f"p = {p_val:.3f}",
            f"n = {len(all_x)} events"
        ]

    # --- Axis Labels ---
    ax.set_xlabel("Bout Duration (s)", fontsize=24)
    ax.set_ylabel(ylabel if ylabel else da_metric, fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=24)

    # --- Y-axis formatting ---
    if ylim:
        ax.set_ylim(ylim)
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        yticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if x.is_integer() else f'{x}'))

    # --- Combined Legend ---
    handles, labels = ax.get_legend_handles_labels()
    stats_label = "\n".join(stats_text_lines)
    stats_handle = plt.Line2D([], [], color='none', label=stats_label)
    handles.append(stats_handle)
    labels.append(stats_label)

    legend = ax.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=16, title='Agent', title_fontsize=18, 
                       frameon=True, facecolor='white', edgecolor='lightgray', fancybox=False)
    legend.get_frame().set_alpha(0.8)

    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    return points_df


