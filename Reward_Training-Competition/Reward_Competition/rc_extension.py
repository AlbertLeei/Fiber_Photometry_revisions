import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import re

from rtc_extension import RTC
from experiment_class import Experiment
from scipy.stats import pearsonr, linregress, ttest_ind
from trial_class import Trial
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
try:
    from figure_settings import apply_plot_style

    apply_plot_style()
except ImportError:
    pass
import seaborn as sns
import pandas as pd
import itertools
import warnings
import pickle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch




class Reward_Competition(RTC):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)

    def save_preprocessed(self, save_path):
        """
        Save the current experiment object so processed trials and derived
        analysis tables can be reused without rerunning the pipeline.
        """
        save_path = os.path.abspath(os.fspath(save_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved Reward_Competition object to {save_path}")
        return save_path

    @classmethod
    def load_preprocessed(cls, save_path):
        """Load a previously saved Reward_Competition object."""
        save_path = os.path.abspath(os.fspath(save_path))
        with open(save_path, "rb") as f:
            exp = pickle.load(f)
        if not isinstance(exp, cls):
            raise TypeError(
                f"Expected a saved {cls.__name__} object, got {type(exp).__name__}."
            )
        print(f"Loaded Reward_Competition object from {save_path}")
        return exp

    """*************************READING CSV AND STORING AS DF**************************"""
    def read_and_merge_manual_scoring(self, csv_file_path):
        """
        1) Reads in and creates a dataframe to store wins and loses and tangles,
        2) Rebuilds file names, looks up trial objects & RTC events,
        3) Builds winner_array, drops unmatched rows.
        """
        # --- (from read_manual_scoring) ---
        df = pd.read_excel(csv_file_path)
        df.columns = df.columns.str.strip().str.lower()
        filtered_columns = (
            df.filter(like='winner')
            .dropna(axis=1, how='all')
            .columns
            .tolist()
        )
        col_to_keep = ['file name', 'subject'] + filtered_columns + ['tangles']
        df = df[col_to_keep]
        self.trials_df = df

        # --- (from merge_data) ---
        # rebuild “file name”
        self.trials_df['file name'] = self.trials_df.apply(
            lambda row: row['subject'] + '-' + '-'.join(row['file name'].split('-')[-2:]),
            axis=1
        )

        # lookup trial object
        self.trials_df['trial'] = self.trials_df.apply(
            lambda row: self.trials.get(row['file name'], None),
            axis=1
        )

        # report mismatches
        print("Total rows:", len(self.trials_df))
        print("Rows with missing trials:", self.trials_df['trial'].isna().sum())

        # extract RTC events
        self.trials_df['sound cues'] = self.trials_df['trial'].apply(
            lambda x: x.rtc_events.get('sound cues', None)
            if isinstance(x, object) and hasattr(x, 'rtc_events') else None
        )
        self.trials_df['port entries'] = self.trials_df['trial'].apply(
            lambda x: x.rtc_events.get('port entries', None)
            if isinstance(x, object) and hasattr(x, 'rtc_events') else None
        )
        self.trials_df['sound cues onset'] = self.trials_df['sound cues'].apply(
            lambda x: x.onset_times if x else None
        )
        self.trials_df['port entries onset'] = self.trials_df['port entries'].apply(
            lambda x: x.onset_times if x else None
        )
        self.trials_df['port entries offset'] = self.trials_df['port entries'].apply(
            lambda x: x.offset_times if x else None
        )

        # subject name column
        self.trials_df['subject_name'] = self.trials_df['file name'].str.split('-').str[0]

        # build winner_array
        winner_columns = [col for col in self.trials_df.columns if 'winner' in col.lower()]
        self.trials_df['winner_array'] = self.trials_df[winner_columns].apply(
            lambda row: row.values.tolist(),
            axis=1
        )
        self.trials_df.drop(columns=winner_columns, inplace=True)

        # drop rows without dopamine data
        self.trials_df.dropna(subset=['trial'], inplace=True)
        self.trials_df.reset_index(drop=True, inplace=True)







    """***********************REMOVING TRIALS WITH TANGLES******************************"""
    def remove_tangles(self, placeholders: bool = False):
        """
        Extracts manual‐scored 'tangles' from self.trials_df and then either
        – fully removes those indices from each per‐event array (placeholders=False),
        – or replaces them with placeholders so that the position of every event
            is preserved (placeholders=True).

        placeholders: bool
        If False (default), drop any tangled slots outright.
        If True, keep the list the same length and insert:
            • None for time‐axes
            • 'tangle' for winner_array
        """

        # 1) parse your original 'tangles' column into a list of ints
        def _parse_tangles(x):
            if pd.isna(x):
                return []
            # x might be a string like "2,5,6"
            return [int(i) for i in str(x).split(',') if i.strip().isdigit()]

        # if you loaded 'tangles' from your manual scoring, parse it
        if 'tangles' in self.trials_df.columns:
            self.trials_df['tangles_array'] = self.trials_df['tangles'].apply(_parse_tangles)
        else:
            # fallback: no manual tangles column
            self.trials_df['tangles_array'] = [[] for _ in range(len(self.trials_df))]

        # 2) define the two strategies
        def _mask(arr, bad_idxs, fill):
            # keep same length, but insert `fill` at each tangled idx
            if not isinstance(arr, (list, np.ndarray)):
                return []
            return [v if i not in bad_idxs else fill
                    for i, v in enumerate(arr)]

        def _prune(arr, bad_idxs, _fill=None):
            # drop any element whose index is in bad_idxs
            if not isinstance(arr, (list, np.ndarray)):
                return []
            return [v for i, v in enumerate(arr) if i not in bad_idxs]

        strategy = _mask if placeholders else _prune

        # 3) apply to each of your columns
        self.trials_df['filtered_winner_array'] = self.trials_df.apply(
            lambda r: strategy(r.get('winner_array', []),
                            r['tangles_array'],
                            'tangle' if placeholders else None),
            axis=1)

        self.trials_df['filtered_sound_cues'] = self.trials_df.apply(
            lambda r: strategy(r.get('sound cues onset', []),
                            r['tangles_array'],
                            np.nan),
            axis=1)

        self.trials_df['filtered_port_entries'] = self.trials_df.apply(
            lambda r: strategy(r.get('port entries onset', []),
                            r['tangles_array'],
                            np.nan),
            axis=1)

        self.trials_df['filtered_port_entry_offset'] = self.trials_df.apply(
            lambda r: strategy(r.get('port entries offset', []),
                            r['tangles_array'],
                            np.nan),
            axis=1)

        # 4) if we pruned (no placeholders), drop any trial with zero events left
        if not placeholders:
            keep_mask = self.trials_df['filtered_sound_cues'].map(len) > 0
            self.trials_df = self.trials_df[keep_mask].reset_index(drop=True)

        # 5) clean up
        self.trials_df.drop(columns=['tangles', 'tangles_array'], errors='ignore', inplace=True)


    """**********************ISOLATING WINNING/LOSING TRIALS************************"""
    def split_by_winner(self, placeholders: bool = False):
        """
        Builds self.winner_df and self.loser_df from self.da_df, excluding ties.
        If placeholders=True, each output list is the same length as the original,
        with np.nan in the slots that were dropped.  Otherwise we prune them out.
        """
        df         = self.da_df.copy()
        id_cols    = ['subject_name', 'file name', 'trial']
        prune_cols = [c for c in df.columns if c not in id_cols]

        def _as_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x if isinstance(x, list) else []

        def _mask_or_prune(seq, wins, subject, keep_win):
            """
            seq      : the actual per-event list (e.g. 'Tone_Zscore', 'Tone_Time_Axis', etc)
            wins     : the filtered_winner_array list
            subject  : the current row's subject_name
            keep_win : True => keep only wins, False => keep only losses
            """
            seq = _as_list(seq)
            wins = _as_list(wins)
            out = []
            for i, val in enumerate(seq):
                w = wins[i] if i < len(wins) else np.nan
                is_win = (not pd.isna(w)) and (w == subject)
                keep   = (keep_win and is_win) or (not keep_win and not is_win)
                if placeholders:
                    out.append(val if keep else np.nan)
                else:
                    if keep:
                        out.append(val)
            return out

        def _build_df(keep_win: bool):
            # apply the mask/prune to every prune_col
            pruned = df.apply(lambda row: pd.Series({
                col: _mask_or_prune(
                        row[col],
                        row['filtered_winner_array'],
                        row['subject_name'],
                        keep_win
                     )
                for col in prune_cols
            }), axis=1)

            full = pd.concat([df[id_cols], pruned], axis=1)

            # drop any sessions that have no kept events
            keycol = prune_cols[0]
            if placeholders:
                # keep session only if at least one non-nan slot remains
                mask = full[keycol].apply(lambda lst: any(not pd.isna(x) for x in lst))
            else:
                # keep session only if list is non-empty
                mask = full[keycol].apply(len) > 0

            return full.loc[mask].reset_index(drop=True)

        # now build both winner_df and loser_df
        self.winner_df = _build_df(keep_win=True)
        self.loser_df  = _build_df(keep_win=False)



    # 2) Build winner/loser/tie splits explicitly (ties = NaN winner slot)
    def split_by_outcome(self, placeholders: bool = False):
        df = self.da_df.copy()
        id_cols = ['subject_name', 'file name', 'trial']
        metric_cols = [c for c in df.columns if c not in id_cols]

        def _as_list(x):
            if isinstance(x, np.ndarray): return x.tolist()
            return x if isinstance(x, list) else []

        def _keep(seq, wins, subject, which, placeholders):
            seq  = _as_list(seq)
            wins = _as_list(wins)
            out = []
            for i, val in enumerate(seq):
                w = wins[i] if i < len(wins) else np.nan
                is_nan  = pd.isna(w)                    # tie
                is_win  = (not is_nan) and (w == subject)
                is_loss = (not is_nan) and (w != subject)
                keep = (which == 'win'  and is_win) or \
                    (which == 'loss' and is_loss) or \
                    (which == 'tie'  and is_nan)
                if placeholders:
                    out.append(val if keep else np.nan)
                else:
                    if keep: out.append(val)
            return out

        def _build(which):
            pruned = df.apply(lambda row: pd.Series({
                col: _keep(row[col],
                        row.get('filtered_winner_array', []),
                        row['subject_name'],
                        which,
                        placeholders)
                for col in metric_cols
            }), axis=1)
            full = pd.concat([df[id_cols], pruned], axis=1)
            key = metric_cols[0]
            if placeholders:
                mask = full[key].apply(lambda lst: any(not pd.isna(x) for x in lst))
            else:
                mask = full[key].apply(lambda lst: isinstance(lst, list) and len(lst) > 0)
            return full.loc[mask].reset_index(drop=True)

        self.winner_df = _build('win')
        self.loser_df  = _build('loss')
        self.tie_df    = _build('tie')





    

    
    """***********************FINDING RANKS******************************"""
    def find_ranks_using_ds(self, file_path):
        def creating_new_df(individuals_array):
            columns = ['ID']
            # Using a loop to add the suffix to each string
            # building columns
            suffix1 = 'w'
            suffix2 = 'm'
            w_array = []
            m_array = []
            for w in individuals_array:
                w_array.append(w + suffix1)
            for m in individuals_array:
                m_array.append(m + suffix2)
            columns.extend(w_array)
            columns.extend(m_array)
            calculations = ['w', 'l', 'w2', 'l2', 'DS']
            columns.extend(calculations)
            # entering base values into the new data frame
            empty_dataframe = pd.DataFrame(columns=columns)
            empty_dataframe['ID'] = individuals_array
            # fill in the data frame with zeros
            pd.set_option('future.no_silent_downcasting', True)
            new_dataframe = empty_dataframe.fillna(0).infer_objects(copy=False)
            new_dataframe[calculations] = new_dataframe[calculations].astype(float)
            return new_dataframe, w_array, m_array
        # Read the Excel file with header
        df = pd.read_excel(file_path, header=0)
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        # Remove unneeded columns
        # to_keep = ['subject_wins', 'other_mouse_wins', 'subject', 'agent']
        to_keep = ['mouse_1_wins', 'mouse_2_wins', 'subject', 'agent']
        # Keep only the specified columns
        df = df[to_keep]
        # splitting matches to find specific individuals
        individuals = set()
        for subjects1 in df['agent']:
                individuals.add(subjects1)
        individuals_list = list(individuals)
        individuals_array = np.array(individuals_list)
        all_individuals_array = np.sort(individuals_array)  # More explicit

        # creating a dataframe using function.
        new_dataframe, w_array, m_array = creating_new_df(all_individuals_array)


        # finding the mice in each match and naming them as mouse 1 or 2 and finding all matches of these mice
        for n, row in df.iterrows():
            
            all_individuals_list = list(all_individuals_array)  # Convert array to list
            mouse1 = df.at[n, 'subject']
            # locate mouse 1 location in all_individuals_array
            index_of_mouse1 = all_individuals_list.index(mouse1)
            mouse2 = df.at[n, 'agent']
            # locate mouse 2 location in all_individuals_array
            index_of_mouse2 = all_individuals_list.index(mouse2)
            mouse1w = df.at[n, 'mouse_1_wins']
            mouse2w = df.at[n, 'mouse_2_wins']
            new_dataframe.loc[index_of_mouse1, mouse2 + 'w'] += int(mouse1w)
            new_dataframe.loc[index_of_mouse2, mouse1 + 'w'] += int(mouse2w)
    
            # total matches between the 2 mice
            mouse_m = mouse1w + mouse2w
            new_dataframe.at[index_of_mouse1, mouse2 + 'm'] += int(mouse_m)
            new_dataframe.at[index_of_mouse2, mouse1 + 'm'] += int(mouse_m)
        

        for i in range(len(all_individuals_array)):
            # w = sum of P(ij) win-rate    P(ij)=(individual1/opponent)
            w = 0
            # l = sum of P(ji) loss-rate   P(ji)=1-(individual1/opponent)
            l = 0
            for j in range(len(all_individuals_array)):
                num_wins = new_dataframe.at[i, w_array[j]]
                num_matches = new_dataframe.at[i, m_array[j]]
                # ensure that there are matches between the two
                if pd.isna(num_matches) or num_matches == 0:
                    continue
                if math.isnan(num_wins):
                    num_wins = 0
                pw = num_wins / num_matches
                pl = 1 - pw
                w += pw
                l += pl
            new_dataframe.loc[i, ['w', 'l']] = float(w), float(l)  # More efficient

            """new_dataframe.at[i, 'w'] = float(w)
            new_dataframe.at[i, 'l'] = float(l)"""

        # calculate the w2 and l2 values
        for i in range(len(all_individuals_array)):
            # w2 = sum of all P(ij) * w(j)
            w2 = 0
            # l2 = sum of all (1-P(ij)) * l(j)
            l2 = 0
            for j in range(len(all_individuals_array)):
                num_wins = new_dataframe.at[i, w_array[j]]
                num_matches = new_dataframe.at[i, m_array[j]]
                # ensure that there are matches between the two
                if math.isnan(num_matches) or num_matches == 0:
                    continue
                if math.isnan(num_wins):
                    num_wins = 0
                pw = num_wins / num_matches
                pl = 1 - pw
                w_opponent = new_dataframe.at[j, 'w']
                l_opponent = new_dataframe.at[j, 'l']
                w2 += pw * w_opponent
                l2 += pl * l_opponent
            new_dataframe.at[i, 'w2'] = float(w2)
            new_dataframe.at[i, 'l2'] = float(l2)

        # calculating David's score = w + w2 - l - l2
        for i in range(len(all_individuals_array)):
            w = new_dataframe.at[i, 'w']
            l = new_dataframe.at[i, 'l']
            w2 = new_dataframe.at[i, 'w2']
            l2 = new_dataframe.at[i, 'l2']
            ds = w + w2 - l - l2
            new_dataframe.at[i, 'DS'] = float(ds)

        # necessary output data
        data_to_keep = ['ID', 'DS']
        new_dataframe = new_dataframe[data_to_keep]

        # for cohort 1 and 2 and combined remove n8
        new_dataframe = new_dataframe.loc[new_dataframe['ID'] != 'n8']  # n8 didn't have any expression

        new_dataframe["Prefix"] = new_dataframe["ID"].str.extract(r"([a-zA-Z]+)")  # Extract letter prefix ('n' or 'p')
        new_dataframe["Number"] = new_dataframe["ID"].str.extract(r"(\d+)").astype(int)  # Extract numeric part

        # Step 3: Assign cages dynamically
        new_dataframe["Cage"] = new_dataframe.apply(lambda row: f"{row['Prefix']}{1 if row['Number'] <= 4 else 2}", axis=1)

        # Step 4: Rank within each cage
        new_dataframe["Rank"] = new_dataframe.groupby("Cage")["DS"].rank(ascending=False, method="min").astype("Int64")

        # Drop helper columns
        new_dataframe = new_dataframe.drop(columns=["Prefix", "Number"])
        return new_dataframe
    
    def merging_ranks(self, rank_df, df=None):
        if df is None:
            df=self.trials_df
        df_merged = df.merge(rank_df, left_on="subject_name", right_on="ID", how="left")

        # Step 2: Drop redundant "ID" column if needed
        df_merged.drop(columns=["ID"], inplace=True)
        df = df_merged
        return df
    


    """*********************** Reading High vs. Low Comp Data******************************"""
    def read_hvl_scoring(self, hvlfp):
        # 1) load and clean up column names
        hv = pd.read_excel(hvlfp)
        hv.columns = hv.columns.str.strip()
        hv = hv.rename(columns={
            'File Name':   'file name',
            'Subject':     'subject_name',
        })

        # 2) identify your PreComp vs Comp columns
        pre_cols  = [c for c in hv.columns if c.lower().endswith('precomp')]
        comp_cols = [c for c in hv.columns if c.lower().endswith('comp') and c not in pre_cols]

        # 3) convert 'T' → NaN → numeric
        hv[pre_cols]  = hv[pre_cols].replace('T', np.nan).apply(pd.to_numeric, errors='coerce')
        hv[comp_cols] = hv[comp_cols].replace('T', np.nan).apply(pd.to_numeric, errors='coerce')

        # 4) pack each row into lists
        hv['HVL_PreComp'] = hv[pre_cols].apply(lambda row: row.tolist(), axis=1)
        hv['HVL_Comp']    = hv[comp_cols].apply(lambda row: row.tolist(), axis=1)

        # 5) rebuild “file name” exactly as in merge_data()
        hv['file name'] = (
            hv['subject_name']
            + '-'
            + hv['file name'].str.split('-').str[-2:].str.join('-')
        )

        # 6) drop any HVL rows not present in trials_df
        valid = self.trials_df[['subject_name','file name']].drop_duplicates()
        hv   = hv.merge(valid, on=['subject_name','file name'], how='inner')

        # 7) now merge those list‐columns into trials_df
        hv = hv[['subject_name','file name','HVL_PreComp','HVL_Comp']]
        self.trials_df = self.trials_df.merge(
            hv,
            on=['subject_name','file name'],
            how='left'
        )

        # 8) fill any missing entries with empty lists
        self.trials_df['HVL_PreComp'] = self.trials_df['HVL_PreComp'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.trials_df['HVL_Comp'] = self.trials_df['HVL_Comp'].apply(
            lambda x: x if isinstance(x, list) else []
        )

    # 1) Keep ties when reading HVL: store a boolean mask, do NOT coerce "T" to NaN
    def read_hvl_scoring_keep_ties(self, hvlfp):
        hv = pd.read_excel(hvlfp)
        hv.columns = hv.columns.str.strip()
        hv = hv.rename(columns={'File Name':'file name','Subject':'subject_name'})

        pre_cols  = [c for c in hv.columns if c.lower().endswith('precomp')]
        comp_cols = [c for c in hv.columns if c.lower().endswith('comp') and c not in pre_cols]

        # Tie masks (True where the cell literally equals "T")
        hv['HVL_PreComp_Tmask'] = hv[pre_cols].apply(lambda r: [v == 'T' for v in r.tolist()], axis=1)
        hv['HVL_Comp_Tmask']    = hv[comp_cols].apply(lambda r: [v == 'T' for v in r.tolist()], axis=1)

        # Also keep numeric versions (turn non-numeric & "T" → NaN) if you need them
        hv_num = hv.copy()
        hv_num[pre_cols]  = hv_num[pre_cols].apply(pd.to_numeric, errors='coerce')
        hv_num[comp_cols] = hv_num[comp_cols].apply(pd.to_numeric, errors='coerce')
        hv['HVL_PreComp'] = hv_num[pre_cols].apply(lambda r: r.tolist(), axis=1)
        hv['HVL_Comp']    = hv_num[comp_cols].apply(lambda r: r.tolist(), axis=1)

        hv['file name'] = hv['subject_name'] + '-' + hv['file name'].str.split('-').str[-2:].str.join('-')

        valid = self.trials_df[['subject_name','file name']].drop_duplicates()
        hv = hv.merge(valid, on=['subject_name','file name'], how='inner')

        keep_cols = ['subject_name','file name','HVL_PreComp','HVL_Comp','HVL_PreComp_Tmask','HVL_Comp_Tmask']
        self.trials_df = self.trials_df.merge(hv[keep_cols], on=['subject_name','file name'], how='left')

        for c in ['HVL_PreComp','HVL_Comp','HVL_PreComp_Tmask','HVL_Comp_Tmask']:
            self.trials_df[c] = self.trials_df[c].apply(lambda x: x if isinstance(x, list) else [])

    def compute_pretrial_EI_DA(self,
                            pretrial_window: tuple[float,float] = (-10.0, 0.0),
                            baseline_window:  tuple[float,float] = (-14.0, -10.0)
                            ) -> pd.DataFrame:
        """
        Compute baseline-corrected peri-event z-score traces for each cue
        over the pretrial period (e.g. –10→0 s relative to each cue).
        Any NaN cues become empty lists so that per-row lengths stay consistent.

        Adds to self.da_df:
        • Pretrial_Time_Axis : list-of-lists of arrays (one per cue)
        • Pretrial_Zscore    : list-of-lists of arrays (one per cue)
        """
        df = self.da_df

        # 1) find the finest dt across all trials
        min_dt = np.inf
        for _, row in df.iterrows():
            ts = np.array(row['trial'].timestamps)
            if ts.size > 1:
                min_dt = min(min_dt, np.min(np.diff(ts)))
        if not np.isfinite(min_dt):
            raise RuntimeError("No valid timestamps found to establish dt.")

        # 2) build common time-axes
        tone_start, tone_end = pretrial_window
        bl_start,   bl_end   = baseline_window

        tone_axis = np.arange(tone_start, tone_end, min_dt)

        # 3) containers
        tone_z, tone_t = [], []
        pe_z,   pe_t   = [], []

        # 4) iterate trials
        for _, row in df.iterrows():
            trial = row['trial']
            ts    = np.array(trial.timestamps)
            zs    = np.array(trial.zscore)

            # get your cues & PEs, defaulting to empty list
            cues = row.get('filtered_sound_cues') or []

            # —— Tone processing —— 
            tz_list, tt_list = [], []
            for i, cue in enumerate(cues):
                # if exactly 40 cues, skip the 40th one entirely - The last one sometimes didn't have enough data so it looked wonky
                if len(cues)==40 and i==39:
                    continue

                # mask out the window around that cue
                mask = (ts >= cue + tone_start) & (ts <= cue + tone_end)
                if not mask.any():
                    tz_list.append(np.full_like(tone_axis, np.nan))
                    tt_list.append(tone_axis.copy())
                    continue

                rel = ts[mask] - cue
                sig = zs[mask]

                # baseline from baseline_window
                blm = (rel >= bl_start) & (rel <= bl_end)
                base = np.nanmean(sig[blm]) if blm.any() else 0.0

                # subtract baseline and re-interpolate
                corr = sig - base
                tz_list.append(np.interp(tone_axis, rel, corr))
                tt_list.append(tone_axis.copy())

            tone_z.append(tz_list)
            tone_t.append(tt_list)

        df['Pretrial_Time_Axis'] = tone_t
        df['Pretrial_Zscore']    = tone_z
        return df

    def compute_competitive_bout_probability_trace(
        self,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        comp_col: str = "HVL_Comp",
        time_window: tuple[float, float] = (-50.0, 55.0),
        bin_size: float = 1.0,
        competitive_values=(1,),
        smoothing_bins: int = 3,
    ):
        """
        Build a cue-aligned probability trace for competitive bouts.

        Because Reward Competition stores competition labels per cue rather than as a
        continuous second-by-second annotation, this method extends each cue's label
        forward until the next cue (or session end) and then aligns that piecewise
        state to every valid tone onset.
        """
        if df is None:
            df = self.da_df

        if cue_col not in df.columns or comp_col not in df.columns:
            raise KeyError(f"Both '{cue_col}' and '{comp_col}' must exist in the dataframe.")

        if bin_size <= 0:
            raise ValueError("bin_size must be > 0.")

        start_s, end_s = time_window
        if end_s <= start_s:
            raise ValueError("time_window must have end > start.")

        rel_time = np.arange(start_s, end_s + bin_size, bin_size, dtype=float)

        numeric_positive = []
        string_positive = set()
        for value in competitive_values:
            try:
                numeric_positive.append(float(value))
            except (TypeError, ValueError):
                string_positive.add(str(value).strip().lower())

        def _scalar_to_binary(value):
            if pd.isna(value):
                return np.nan

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                label = str(value).strip().lower()
                return 1.0 if label in string_positive else 0.0

            if any(np.isclose(numeric_value, target) for target in numeric_positive):
                return 1.0
            return 0.0

        aligned_rows = []
        contributing_sessions = 0

        for _, row in df.iterrows():
            raw_cues = row.get(cue_col, [])
            raw_states = row.get(comp_col, [])

            cues = np.asarray(raw_cues if isinstance(raw_cues, (list, np.ndarray)) else [], dtype=float)
            if cues.size == 0:
                continue

            states_list = raw_states if isinstance(raw_states, (list, np.ndarray)) else []
            n_pairs = min(len(cues), len(states_list))
            if n_pairs == 0:
                continue

            cues = cues[:n_pairs]
            states = np.asarray([_scalar_to_binary(value) for value in states_list[:n_pairs]], dtype=float)

            valid_cue_mask = np.isfinite(cues)
            if not np.any(valid_cue_mask):
                continue

            cues = cues[valid_cue_mask]
            states = states[valid_cue_mask]

            trial_obj = row.get("trial", None)
            timestamps = np.asarray(getattr(trial_obj, "timestamps", []), dtype=float)
            if timestamps.size > 0:
                session_start = float(np.nanmin(timestamps))
                session_end = float(np.nanmax(timestamps))
            else:
                session_start = float(np.nanmin(cues))
                session_end = float(np.nanmax(cues))

            valid_anchor_mask = np.isfinite(states)
            if not np.any(valid_anchor_mask):
                continue

            contributing_sessions += 1

            for anchor_cue in cues[valid_anchor_mask]:
                absolute_times = anchor_cue + rel_time
                interval_idx = np.searchsorted(cues, absolute_times, side="right") - 1

                sample = np.full(rel_time.shape, np.nan, dtype=float)
                in_bounds = (
                    (interval_idx >= 0)
                    & (absolute_times >= session_start)
                    & (absolute_times <= session_end)
                )
                if np.any(in_bounds):
                    sample[in_bounds] = states[interval_idx[in_bounds]]

                aligned_rows.append(sample)

        if not aligned_rows:
            raise ValueError("No valid cue-aligned competitive bouts were found for plotting.")

        aligned = np.vstack(aligned_rows)
        proportion_active = np.nanmean(aligned, axis=0)
        percent_active = proportion_active * 100.0
        contributing_n = np.sum(np.isfinite(aligned), axis=0)

        if aligned.shape[0] > 1:
            sem = np.nanstd(aligned, axis=0, ddof=1) / np.sqrt(np.maximum(contributing_n, 1))
        else:
            sem = np.full(rel_time.shape, np.nan, dtype=float)

        percent_sem = sem * 100.0

        if smoothing_bins and smoothing_bins > 1:
            proportion_active = (
                pd.Series(proportion_active)
                .rolling(window=int(smoothing_bins), center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            percent_active = proportion_active * 100.0

        summary_df = pd.DataFrame(
            {
                "time_s": rel_time,
                "proportion_active": proportion_active,
                "percent_active": percent_active,
                "sem": sem,
                "percent_sem": percent_sem,
                "n_contributing": contributing_n,
            }
        )

        return summary_df, aligned, {
            "n_trials": int(aligned.shape[0]),
            "n_sessions": int(contributing_sessions),
            "cue_col": cue_col,
            "comp_col": comp_col,
        }

    def plot_competitive_bout_probability_over_session(
        self,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        comp_col: str = "HVL_Comp",
        time_window: tuple[float, float] = (-50.0, 55.0),
        bin_size: float = 1.0,
        competitive_values=(1,),
        smoothing_bins: int = 3,
        tone_window: tuple[float, float] = (0.0, 10.0),
        reward_window: tuple[float, float] = (4.0, 6.0),
        figsize: tuple = (5.6, 3.0),
        trace_color: str = "#222222",
        trace_shadow_color: str = "#8A8A8A",
        tone_color: str = "#F4A300",
        reward_color: str = "#C8102E",
        y_mode: str = "proportion",
        ylabel: str = None,
        title: str = None,
        ylim: tuple = (0.2, 0.8),
        ax=None,
        save_path: str = None,
    ):
        """
        Plot the cue-aligned competitive-bout probability trace in the same style
        as the Home Cage session-duration plot.
        """
        summary_df, aligned, meta = self.compute_competitive_bout_probability_trace(
            df=df,
            cue_col=cue_col,
            comp_col=comp_col,
            time_window=time_window,
            bin_size=bin_size,
            competitive_values=competitive_values,
            smoothing_bins=smoothing_bins,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if tone_window is not None:
            ax.axvspan(
                tone_window[0],
                tone_window[1],
                facecolor=tone_color,
                edgecolor=tone_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.35,
                zorder=0,
            )

        if reward_window is not None:
            ax.axvspan(
                reward_window[0],
                reward_window[1],
                facecolor=reward_color,
                edgecolor=reward_color,
                alpha=0.55,
                zorder=1,
            )

        if y_mode == "proportion":
            y_col = "proportion_active"
            default_ylabel = "Competitive bout active"
        elif y_mode == "percent":
            y_col = "percent_active"
            default_ylabel = "% competitive bout active"
        else:
            raise ValueError("y_mode must be either 'proportion' or 'percent'.")

        x = summary_df["time_s"].to_numpy()
        y = summary_df[y_col].to_numpy()

        ax.plot(x, y, color=trace_shadow_color, linewidth=3.8, alpha=0.45, zorder=2)
        ax.plot(x, y, color=trace_color, linewidth=1.8, zorder=3)

        ax.set_xlabel("Time relative to tone onset (s)", fontsize=11)
        ax.set_ylabel(ylabel or default_ylabel, fontsize=11)
        if title:
            ax.set_title(title, fontsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlim(time_window[0], time_window[1])
        xticks = np.arange(np.ceil(time_window[0] / 10) * 10, time_window[1] + 1, 10)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(tick)}" for tick in xticks], rotation=35, ha="right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend_handles = []
        if tone_window is not None:
            legend_handles.append(Patch(facecolor=tone_color, label="Tone window"))
        if reward_window is not None:
            legend_handles.append(Patch(facecolor=reward_color, label="Reward window"))
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title=f"trials={meta['n_trials']}",
                loc="upper right",
                frameon=True,
                fontsize=9,
                title_fontsize=9,
            )

        if save_path:
            ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

        return ax, summary_df, aligned

    def compute_binned_competitive_bout_activity(
        self,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        comp_col: str = "HVL_Comp",
        time_window: tuple[float, float] = (-50.0, 55.0),
        sample_step_s: float = 1.0,
        bin_width_s: float = 5.0,
        competitive_values=(1,),
    ):
        """
        Create per-trial 0/1 vectors across time bins, then average across trials.

        For each aligned trial and each time bin:
        - 1 if a competitive bout is active anywhere in that bin
        - 0 if the bin is observed and no competitive bout is active
        - NaN if the bin is entirely unobserved for that trial
        """
        if bin_width_s <= 0:
            raise ValueError("bin_width_s must be > 0.")
        if sample_step_s <= 0:
            raise ValueError("sample_step_s must be > 0.")
        if bin_width_s < sample_step_s:
            warnings.warn(
                "bin_width_s is smaller than sample_step_s, so many bins may contain no samples. "
                "Use sample_step_s <= bin_width_s for meaningful binning."
            )

        trace_summary_df, aligned, meta = self.compute_competitive_bout_probability_trace(
            df=df,
            cue_col=cue_col,
            comp_col=comp_col,
            time_window=time_window,
            bin_size=sample_step_s,
            competitive_values=competitive_values,
            smoothing_bins=1,
        )

        rel_time = trace_summary_df["time_s"].to_numpy(dtype=float)
        start_s, end_s = time_window

        bin_edges = np.arange(start_s, end_s + bin_width_s, bin_width_s, dtype=float)
        if bin_edges[-1] < end_s:
            bin_edges = np.append(bin_edges, end_s)

        n_bins = len(bin_edges) - 1
        binned_trials = np.full((aligned.shape[0], n_bins), np.nan, dtype=float)

        for bin_idx in range(n_bins):
            left = bin_edges[bin_idx]
            right = bin_edges[bin_idx + 1]
            if bin_idx == n_bins - 1:
                mask = (rel_time >= left) & (rel_time <= right)
            else:
                mask = (rel_time >= left) & (rel_time < right)

            if not np.any(mask):
                continue

            bin_slice = aligned[:, mask]
            observed_mask = np.any(np.isfinite(bin_slice), axis=1)
            active_mask = np.any(bin_slice == 1, axis=1)

            binned_trials[observed_mask, bin_idx] = active_mask[observed_mask].astype(float)

        n_contributing_trials = np.sum(np.isfinite(binned_trials), axis=0)
        proportion_active = np.nanmean(binned_trials, axis=0)
        percent_active = proportion_active * 100.0

        if binned_trials.shape[0] > 1:
            sem = np.nanstd(binned_trials, axis=0, ddof=1) / np.sqrt(np.maximum(n_contributing_trials, 1))
        else:
            sem = np.full(n_bins, np.nan, dtype=float)

        percent_sem = sem * 100.0
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0

        binned_summary_df = pd.DataFrame(
            {
                "bin_start_s": bin_edges[:-1],
                "bin_end_s": bin_edges[1:],
                "bin_center_s": bin_centers,
                "bin_width_s": np.diff(bin_edges),
                "proportion_active": proportion_active,
                "percent_active": percent_active,
                "sem": sem,
                "percent_sem": percent_sem,
                "n_contributing_trials": n_contributing_trials,
            }
        )

        meta = dict(meta)
        meta["bin_width_s"] = float(bin_width_s)
        meta["sample_step_s"] = float(sample_step_s)

        return binned_summary_df, binned_trials, aligned, meta

    def plot_binned_competitive_bout_activity(
        self,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        comp_col: str = "HVL_Comp",
        time_window: tuple[float, float] = (-50.0, 55.0),
        sample_step_s: float = 1.0,
        bin_width_s: float = 5.0,
        competitive_values=(1,),
        y_mode: str = "percent",
        plot_style: str = "step",
        tone_window: tuple[float, float] = (0.0, 10.0),
        reward_window: tuple[float, float] = (4.0, 6.0),
        figsize: tuple = (5.8, 3.2),
        trace_color: str = "#222222",
        trace_shadow_color: str = "#8A8A8A",
        tone_color: str = "#F4A300",
        reward_color: str = "#C8102E",
        ylabel: str = None,
        title: str = None,
        ylim: tuple = None,
        ax=None,
        save_path: str = None,
    ):
        """
        Plot binarized competitive-bout activity averaged across aligned trials.
        """
        summary_df, binned_trials, aligned, meta = self.compute_binned_competitive_bout_activity(
            df=df,
            cue_col=cue_col,
            comp_col=comp_col,
            time_window=time_window,
            sample_step_s=sample_step_s,
            bin_width_s=bin_width_s,
            competitive_values=competitive_values,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if tone_window is not None:
            ax.axvspan(
                tone_window[0],
                tone_window[1],
                facecolor=tone_color,
                edgecolor=tone_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.35,
                zorder=0,
            )

        if reward_window is not None:
            ax.axvspan(
                reward_window[0],
                reward_window[1],
                facecolor=reward_color,
                edgecolor=reward_color,
                alpha=0.55,
                zorder=1,
            )

        if y_mode == "proportion":
            y_col = "proportion_active"
            default_ylabel = "Fraction of trials with comp bout active"
        elif y_mode == "percent":
            y_col = "percent_active"
            default_ylabel = "% trials with comp bout active"
        else:
            raise ValueError("y_mode must be either 'proportion' or 'percent'.")

        x = summary_df["bin_center_s"].to_numpy()
        y = summary_df[y_col].to_numpy()
        bin_width = float(summary_df["bin_width_s"].iloc[0]) if len(summary_df) else bin_width_s

        if plot_style == "bar":
            ax.bar(x, y, width=bin_width * 0.9, color=trace_shadow_color, edgecolor=trace_color, linewidth=1.2, zorder=2)
        elif plot_style == "step":
            ax.step(x, y, where="mid", color=trace_shadow_color, linewidth=4.0, alpha=0.35, zorder=2)
            ax.step(x, y, where="mid", color=trace_color, linewidth=2.0, zorder=3)
            ax.plot(x, y, "o", color=trace_color, markersize=4, zorder=4)
        else:
            raise ValueError("plot_style must be either 'step' or 'bar'.")

        ax.set_xlabel("Time relative to tone onset (s)", fontsize=11)
        ax.set_ylabel(ylabel or default_ylabel, fontsize=11)
        if title:
            ax.set_title(title, fontsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlim(time_window[0], time_window[1])
        xticks = np.arange(np.ceil(time_window[0] / 10) * 10, time_window[1] + 1, 10)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(tick)}" for tick in xticks], rotation=35, ha="right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            handles=[
                Patch(facecolor=tone_color, label="Tone window"),
                Patch(facecolor=reward_color, label="Reward window"),
            ],
            title=f"trials={meta['n_trials']}, bins={meta['bin_width_s']:.0f}s",
            loc="upper right",
            frameon=True,
            fontsize=9,
            title_fontsize=9,
        )

        if save_path:
            ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

        return ax, summary_df, binned_trials, aligned

    def compute_session_trial_competitive_bout_metric(
        self,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        event_col: str = "filtered_winner_array",
        deduplicate_sessions: bool = True,
        session_id_col: str = "shared_session_id",
    ):
        """
        Treat each tone-to-next-tone interval as one full trial, including ITI.

        For each session and trial index:
        - comp_bout_count = 1 if any competitive bout is coded for that trial
          interval, otherwise 0
        - comp_bout_frequency_hz = count / interval_duration_s
        - comp_bout_frequency_per_min = count / interval_duration_min

        Notes
        -----
        With the current RC table structure, `event_col` is interpreted as one
        competition code per trial interval. So this produces a 0/1 count per
        interval. If later you add actual multiple bout timestamps per interval,
        this function can be extended to count more than one bout.
        """
        if df is None:
            df = self.da_df.copy()
        else:
            df = df.copy()

        required_cols = ["file name", cue_col, event_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        def _shared_session_id(file_name):
            if pd.isna(file_name):
                return None
            parts = str(file_name).split("-", 1)
            return parts[1] if len(parts) > 1 else str(file_name)

        def _as_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x if isinstance(x, list) else []

        def _event_to_count(value):
            if pd.isna(value):
                return 0.0
            if isinstance(value, str):
                clean = value.strip().lower()
                if clean in {"", "nan", "none", "tangle"}:
                    return 0.0
                return 1.0
            return 1.0

        df[session_id_col] = df["file name"].map(_shared_session_id)

        if deduplicate_sessions:
            df = df.drop_duplicates(subset=[session_id_col], keep="first").reset_index(drop=True)

        trial_rows = []

        for _, row in df.iterrows():
            session_id = row[session_id_col]
            file_name = row["file name"]
            cues = np.asarray(_as_list(row.get(cue_col, [])), dtype=float)
            events = _as_list(row.get(event_col, []))

            cues = cues[np.isfinite(cues)]
            if len(cues) < 2:
                continue

            n_trials = min(len(cues) - 1, len(events))
            if n_trials <= 0:
                continue

            for trial_idx in range(n_trials):
                start_s = float(cues[trial_idx])
                end_s = float(cues[trial_idx + 1])
                duration_s = end_s - start_s
                if not np.isfinite(duration_s) or duration_s <= 0:
                    continue

                count = float(_event_to_count(events[trial_idx]))
                frequency_hz = count / duration_s
                frequency_per_min = count / (duration_s / 60.0)

                trial_rows.append(
                    {
                        session_id_col: session_id,
                        "file name": file_name,
                        "trial_number": int(trial_idx + 1),
                        "trial_start_s": start_s,
                        "trial_end_s": end_s,
                        "trial_duration_s": duration_s,
                        "comp_bout_count": count,
                        "comp_bout_frequency_hz": frequency_hz,
                        "comp_bout_frequency_per_min": frequency_per_min,
                    }
                )

        trial_df = pd.DataFrame(trial_rows)
        if trial_df.empty:
            raise ValueError("No valid tone-to-tone trial intervals were found.")

        def _sem(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            if len(x) <= 1:
                return np.nan
            return np.std(x, ddof=1) / np.sqrt(len(x))

        summary_df = (
            trial_df.groupby("trial_number", as_index=False)
            .agg(
                mean_comp_bout_count=("comp_bout_count", "mean"),
                sem_comp_bout_count=("comp_bout_count", _sem),
                mean_comp_bout_frequency_hz=("comp_bout_frequency_hz", "mean"),
                sem_comp_bout_frequency_hz=("comp_bout_frequency_hz", _sem),
                mean_comp_bout_frequency_per_min=("comp_bout_frequency_per_min", "mean"),
                sem_comp_bout_frequency_per_min=("comp_bout_frequency_per_min", _sem),
                mean_trial_duration_s=("trial_duration_s", "mean"),
                n_sessions=("comp_bout_count", "count"),
            )
        )

        return summary_df, trial_df

    def plot_session_trial_competitive_bout_metric(
        self,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        event_col: str = "filtered_winner_array",
        metric: str = "frequency_per_min",
        deduplicate_sessions: bool = True,
        figsize: tuple = (6.2, 3.6),
        color: str = "#222222",
        error_color: str = "#8A8A8A",
        marker: str = "o",
        linewidth: float = 2.0,
        capsize: float = 3.0,
        ylabel: str = None,
        title: str = None,
        ylim: tuple = None,
        ax=None,
        save_path: str = None,
    ):
        """
        Plot session-averaged competitive-bout count or frequency by trial number.
        """
        summary_df, trial_df = self.compute_session_trial_competitive_bout_metric(
            df=df,
            cue_col=cue_col,
            event_col=event_col,
            deduplicate_sessions=deduplicate_sessions,
        )

        if metric == "count":
            y_col = "mean_comp_bout_count"
            err_col = "sem_comp_bout_count"
            default_ylabel = "Competitive bout count"
        elif metric == "frequency_hz":
            y_col = "mean_comp_bout_frequency_hz"
            err_col = "sem_comp_bout_frequency_hz"
            default_ylabel = "Competitive bout frequency (Hz)"
        elif metric == "frequency_per_min":
            y_col = "mean_comp_bout_frequency_per_min"
            err_col = "sem_comp_bout_frequency_per_min"
            default_ylabel = "Competitive bout frequency (/min)"
        else:
            raise ValueError("metric must be 'count', 'frequency_hz', or 'frequency_per_min'.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        x = summary_df["trial_number"].to_numpy()
        y = summary_df[y_col].to_numpy()
        yerr = summary_df[err_col].to_numpy()

        ax.plot(x, y, color=error_color, linewidth=linewidth + 2, alpha=0.25, zorder=1)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            ecolor=error_color,
            linewidth=linewidth,
            marker=marker,
            markersize=5,
            capsize=capsize,
            zorder=3,
        )

        ax.set_xlabel("Trial number", fontsize=11)
        ax.set_ylabel(ylabel or default_ylabel, fontsize=11)
        if title:
            ax.set_title(title, fontsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if save_path:
            ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

        return ax, summary_df, trial_df

    def load_competition_bout_csv_folder(
        self,
        csv_folder: str,
        behavior_name: str = "Competition Bout",
        merge_touching_bouts: bool = True,
        session_id_col: str = "shared_session_id",
    ):
        """
        Load timestamped competition bouts from a folder of BORIS-style CSVs.

        Files are grouped by shared recording/session id so duplicated subject-level
        exports can be collapsed into one session-level set of competitive bouts.
        Overlapping bouts across files are merged to avoid double counting.
        """
        csv_paths = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in: {csv_folder}")

        all_rows = []

        def _shared_session_id(file_stem):
            parts = str(file_stem).split("-", 1)
            return parts[1] if len(parts) > 1 else str(file_stem)

        for path in csv_paths:
            try:
                bout_df = pd.read_csv(path)
            except Exception:
                continue

            if "Behavior" not in bout_df.columns or "Start (s)" not in bout_df.columns or "Stop (s)" not in bout_df.columns:
                continue

            bout_df = bout_df.copy()
            bout_df["Start (s)"] = pd.to_numeric(bout_df["Start (s)"], errors="coerce")
            bout_df["Stop (s)"] = pd.to_numeric(bout_df["Stop (s)"], errors="coerce")
            bout_df = bout_df[
                (bout_df["Behavior"].astype(str).str.strip() == behavior_name)
                & np.isfinite(bout_df["Start (s)"])
                & np.isfinite(bout_df["Stop (s)"])
                & (bout_df["Stop (s)"] > bout_df["Start (s)"])
            ].copy()

            if bout_df.empty:
                continue

            file_stem = os.path.splitext(os.path.basename(path))[0]
            bout_df["csv_file_name"] = file_stem
            bout_df[session_id_col] = _shared_session_id(file_stem)
            all_rows.append(bout_df)

        if not all_rows:
            raise ValueError(f"No '{behavior_name}' rows with valid timestamps were found in {csv_folder}.")

        raw_bout_df = pd.concat(all_rows, ignore_index=True)

        merged_rows = []
        for session_id, group in raw_bout_df.groupby(session_id_col):
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
                            session_id_col: session_id,
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
                    session_id_col: session_id,
                    "merged_bout_start_s": float(current_start),
                    "merged_bout_stop_s": float(current_stop),
                    "merged_bout_duration_s": float(current_stop - current_start),
                    "merged_source_rows": int(merged_count),
                }
            )

        merged_bout_df = pd.DataFrame(merged_rows)
        if merged_bout_df.empty:
            raise ValueError("No merged competitive bouts could be constructed from the CSV folder.")

        return raw_bout_df, merged_bout_df

    def compute_csv_session_trial_competitive_bout_metric(
        self,
        csv_folder: str,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        precomp_col: str = "HVL_PreComp",
        comp_col: str = "HVL_Comp",
        pretrial_window: tuple[float, float] = (-10.0, 0.0),
        behavior_name: str = "Competition Bout",
        include_pretrial: bool = True,
        count_mode: str = "start",
        session_id_col: str = "shared_session_id",
    ):
        """
        Compute session-level competitive bout count/frequency by trial number using
        timestamped bouts from full-data CSVs.

        Window types:
        - `trial`: tone_i -> tone_{i+1}` including the ITI
        - `pretrial`: fixed window before tone_i, default -10 -> 0 s

        Counts are based on merged competitive bouts collapsed across subject and
        social-agent CSV exports for the same recording session.
        """
        if df is None:
            if not hasattr(self, "da_df") or self.da_df is None or len(self.da_df) == 0:
                raise ValueError("No da_df available. Run the RC preprocessing pipeline first.")
            df = self.da_df.copy()
        else:
            df = df.copy()

        raw_bout_df, merged_bout_df = self.load_competition_bout_csv_folder(
            csv_folder=csv_folder,
            behavior_name=behavior_name,
            session_id_col=session_id_col,
        )

        def _shared_session_id(file_name):
            if pd.isna(file_name):
                return None
            parts = str(file_name).split("-", 1)
            return parts[1] if len(parts) > 1 else str(file_name)

        def _as_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x if isinstance(x, list) else []

        df[session_id_col] = df["file name"].map(_shared_session_id)
        session_df = df.drop_duplicates(subset=[session_id_col], keep="first").reset_index(drop=True)

        merged_sessions = set(merged_bout_df[session_id_col].dropna().unique())
        session_df = session_df[session_df[session_id_col].isin(merged_sessions)].reset_index(drop=True)

        if session_df.empty:
            raise ValueError("No overlap was found between da_df sessions and the CSV folder sessions.")

        def _count_bouts_in_window(bout_group, start_s, end_s):
            if count_mode == "start":
                mask = (bout_group["merged_bout_start_s"] >= start_s) & (bout_group["merged_bout_start_s"] < end_s)
            elif count_mode == "overlap":
                mask = (bout_group["merged_bout_stop_s"] > start_s) & (bout_group["merged_bout_start_s"] < end_s)
            else:
                raise ValueError("count_mode must be 'start' or 'overlap'.")
            return float(mask.sum())

        metric_rows = []

        for _, row in session_df.iterrows():
            session_id = row[session_id_col]
            file_name = row["file name"]
            cues = np.asarray(_as_list(row.get(cue_col, [])), dtype=float)
            cues = cues[np.isfinite(cues)]
            if len(cues) == 0:
                continue

            bout_group = merged_bout_df[merged_bout_df[session_id_col] == session_id].copy()
            if bout_group.empty:
                continue

            precomp_labels = _as_list(row.get(precomp_col, []))
            comp_labels = _as_list(row.get(comp_col, []))

            if include_pretrial:
                n_pre = min(len(cues), len(precomp_labels)) if len(precomp_labels) > 0 else len(cues)
                for trial_idx in range(n_pre):
                    tone_s = float(cues[trial_idx])
                    start_s = tone_s + float(pretrial_window[0])
                    end_s = tone_s + float(pretrial_window[1])
                    duration_s = end_s - start_s
                    if duration_s <= 0:
                        continue

                    count = _count_bouts_in_window(bout_group, start_s, end_s)
                    metric_rows.append(
                        {
                            session_id_col: session_id,
                            "file name": file_name,
                            "window_type": "pretrial",
                            "trial_number": int(trial_idx + 1),
                            "window_start_s": start_s,
                            "window_end_s": end_s,
                            "window_duration_s": duration_s,
                            "comp_bout_count": count,
                            "comp_bout_frequency_hz": count / duration_s,
                            "comp_bout_frequency_per_min": count / (duration_s / 60.0),
                        }
                    )

            if len(cues) >= 2:
                n_trial = min(len(cues) - 1, len(comp_labels)) if len(comp_labels) > 0 else len(cues) - 1
                for trial_idx in range(n_trial):
                    start_s = float(cues[trial_idx])
                    end_s = float(cues[trial_idx + 1])
                    duration_s = end_s - start_s
                    if duration_s <= 0:
                        continue

                    count = _count_bouts_in_window(bout_group, start_s, end_s)
                    metric_rows.append(
                        {
                            session_id_col: session_id,
                            "file name": file_name,
                            "window_type": "trial",
                            "trial_number": int(trial_idx + 1),
                            "window_start_s": start_s,
                            "window_end_s": end_s,
                            "window_duration_s": duration_s,
                            "comp_bout_count": count,
                            "comp_bout_frequency_hz": count / duration_s,
                            "comp_bout_frequency_per_min": count / (duration_s / 60.0),
                        }
                    )

        metric_df = pd.DataFrame(metric_rows)
        if metric_df.empty:
            raise ValueError("No trial or pretrial competitive bout metrics could be computed from the CSV folder.")

        def _sem(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            if len(x) <= 1:
                return np.nan
            return np.std(x, ddof=1) / np.sqrt(len(x))

        summary_df = (
            metric_df.groupby(["window_type", "trial_number"], as_index=False)
            .agg(
                mean_comp_bout_count=("comp_bout_count", "mean"),
                sem_comp_bout_count=("comp_bout_count", _sem),
                mean_comp_bout_frequency_hz=("comp_bout_frequency_hz", "mean"),
                sem_comp_bout_frequency_hz=("comp_bout_frequency_hz", _sem),
                mean_comp_bout_frequency_per_min=("comp_bout_frequency_per_min", "mean"),
                sem_comp_bout_frequency_per_min=("comp_bout_frequency_per_min", _sem),
                mean_window_duration_s=("window_duration_s", "mean"),
                n_sessions=("comp_bout_count", "count"),
            )
        )

        return summary_df, metric_df, raw_bout_df, merged_bout_df

    def plot_csv_session_trial_competitive_bout_metric(
        self,
        csv_folder: str,
        df: pd.DataFrame = None,
        cue_col: str = "filtered_sound_cues",
        precomp_col: str = "HVL_PreComp",
        comp_col: str = "HVL_Comp",
        pretrial_window: tuple[float, float] = (-10.0, 0.0),
        behavior_name: str = "Competition Bout",
        window_type: str = "trial",
        metric: str = "frequency_per_min",
        include_pretrial: bool = True,
        count_mode: str = "start",
        figsize: tuple = (6.4, 3.8),
        color: str = "#222222",
        error_color: str = "#8A8A8A",
        marker: str = "o",
        linewidth: float = 2.0,
        capsize: float = 3.0,
        ylabel: str = None,
        title: str = None,
        ylim: tuple = None,
        ax=None,
    ):
        """
        Plot session-averaged competitive bout count/frequency by trial number
        using timestamped competition-bout CSVs.
        """
        summary_df, metric_df, raw_bout_df, merged_bout_df = self.compute_csv_session_trial_competitive_bout_metric(
            csv_folder=csv_folder,
            df=df,
            cue_col=cue_col,
            precomp_col=precomp_col,
            comp_col=comp_col,
            pretrial_window=pretrial_window,
            behavior_name=behavior_name,
            include_pretrial=include_pretrial,
            count_mode=count_mode,
        )

        plot_df = summary_df[summary_df["window_type"] == window_type].copy()
        if plot_df.empty:
            raise ValueError(f"No rows found for window_type='{window_type}'.")

        if metric == "count":
            y_col = "mean_comp_bout_count"
            err_col = "sem_comp_bout_count"
            default_ylabel = "Competitive bout count"
        elif metric == "frequency_hz":
            y_col = "mean_comp_bout_frequency_hz"
            err_col = "sem_comp_bout_frequency_hz"
            default_ylabel = "Competitive bout frequency (Hz)"
        elif metric == "frequency_per_min":
            y_col = "mean_comp_bout_frequency_per_min"
            err_col = "sem_comp_bout_frequency_per_min"
            default_ylabel = "Competitive bout frequency (/min)"
        else:
            raise ValueError("metric must be 'count', 'frequency_hz', or 'frequency_per_min'.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        x = plot_df["trial_number"].to_numpy()
        y = plot_df[y_col].to_numpy()
        yerr = plot_df[err_col].to_numpy()

        ax.plot(x, y, color=error_color, linewidth=linewidth + 2, alpha=0.25, zorder=1)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            ecolor=error_color,
            linewidth=linewidth,
            marker=marker,
            markersize=5,
            capsize=capsize,
            zorder=3,
        )

        ax.set_xlabel("Trial number", fontsize=11)
        ax.set_ylabel(ylabel or default_ylabel, fontsize=11)
        if title:
            ax.set_title(title, fontsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax, plot_df, metric_df, raw_bout_df, merged_bout_df















    """********************** PLOTTING PSTH******************************"""
    # plots EI peth for every event
    def rc_plot_peth_per_event(self, df, i, directory_path, title='PETH graph for n trials', signal_type='zscore', 
                            error_type='sem', display_pre_time=4, display_post_time=10, yticks_interval=2):
        
        # Plots the PETH for each event index (e.g., each sound cue) across all trials in one figure with subplots.
        
        if df is None:
            df = self.trials_df
        # Determine the indices for the display range
        time_axis = df.iloc[i]['Tone Event_Time_Axis'][0]
        display_start_idx = np.searchsorted(time_axis, -display_pre_time)
        display_end_idx = np.searchsorted(time_axis, display_post_time)
        time_axis = time_axis[display_start_idx:display_end_idx]

        num_events = 19

        # Create subplots arranged horizontally
        fig, axes = plt.subplots(1, num_events, figsize=(5 * num_events, 5), sharey=True)
        print(len(df.iloc[i]['Tone Event_Zscore']))
        for idx, event_index in enumerate(range(num_events)):
            ax = axes[idx]

            tone_event_zscores = df.iloc[i]['Tone Event_Zscore']

            # Ensure it's a list/array and has enough elements
            if not isinstance(tone_event_zscores, (list, np.ndarray)) or len(tone_event_zscores) == 0:
                print(f"Skipping row {i}, 'Tone Event_Zscore' is empty or NaN.")
                continue

            if event_index >= len(tone_event_zscores):
                print(f"Skipping index {event_index}, out of range for row {i}.")
                continue  

            event_traces = tone_event_zscores[event_index]  # Safe access
            
            # Check if event_traces is empty
            if event_traces.size == 0:
                print(f"No data for event {event_index + 1}")
                continue
            
            # Convert list of traces to numpy array
            event_traces = np.array(event_traces)
            
            # Ensure event_traces is 2D (if only one trace, add a new axis to make it 2D)
            if event_traces.ndim == 1:
                event_traces = np.expand_dims(event_traces, axis=0)  # Convert 1D array to 2D
            
            # Truncate the traces to the display range
            event_traces = event_traces[:, display_start_idx:display_end_idx]

            # Plot the traces (without error bars)
            ax.plot(time_axis, event_traces.T, color='b', label=f'Event {event_index + 1}', linewidth=1.5)  # Transpose for proper plotting
            ax.axvline(0, color='black', linestyle='--', label='Event onset')

            # Set the x-ticks to show only the last time, 0, and the very end time
            ax.set_xticks([time_axis[0], 0, time_axis[-1]])
            ax.set_xticklabels([f'{time_axis[0]:.1f}', '0', f'{time_axis[-1]:.1f}'], fontsize=12)

            # Set the y-tick labels with specified interval
            if idx == 0:
                y_min, y_max = ax.get_ylim()
                y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval,
                                    np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval,
                                    yticks_interval)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=12)
            else:
                ax.set_yticks([])  # Hide y-ticks for other subplots

            ax.set_xlabel('Time (s)', fontsize=14)
            if idx == 0:
                ax.set_ylabel(f'{signal_type.capitalize()} dFF', fontsize=14)

            # Set the title for each event
            ax.set_title(f'Event {event_index + 1}', fontsize=14)

            # Remove the right and top spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Adjust layout and add a common title
        plt.suptitle(title, fontsize=16)
        save_path = os.path.join(str(directory_path) + '\\' + str(i) + 'all_PETH.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


    def apply_rc_plot_peth(self, df, directory_path, title='PETH graph for n trials', 
                       signal_type='zscore', error_type='sem', display_pre_time=4, 
                       display_post_time=10, yticks_interval=2):
        
        # Applies the rc_plot_peth_per_event function to all rows in the DataFrame.
        
        # Loop through all rows in the DataFrame
        for i, row in df.iterrows():
            # Call the plotting function for each row
            self.rc_plot_peth_per_event(
                df, 
                i, 
                directory_path,
                title=title, 
                signal_type=signal_type, 
                error_type=error_type, 
                display_pre_time=display_pre_time, 
                display_post_time=display_post_time, 
                yticks_interval=yticks_interval
            )


        
    """********************************MISC*************************************"""
    def downsample_data(self, data, time_axis, bin_size=10):
            """
            Downsamples the time series data by averaging over bins of 'bin_size' points.
            
            Parameters:
            - data (1D NumPy array): Original Z-score values.
            - time_axis (1D NumPy array): Corresponding time points.
            - bin_size (int): Number of original bins to merge into one.
            
            Returns:
            - downsampled_data (1D NumPy array): Smoothed Z-score values.
            - new_time_axis (1D NumPy array): Adjusted time points.
            """
            num_bins = len(data) // bin_size  # Number of new bins
            data = data[:num_bins * bin_size]  # Trim excess points
            time_axis = time_axis[:num_bins * bin_size]

            # Reshape and compute mean for each bin
            downsampled_data = data.reshape(num_bins, bin_size).mean(axis=1)
            new_time_axis = time_axis.reshape(num_bins, bin_size).mean(axis=1)

            return downsampled_data, new_time_axis


    def compute_subject_mean_traces(self,
                                    df: pd.DataFrame,
                                    event_type: str) -> pd.DataFrame:
        """
        For each subject in `df`:
        1) collapse all f"{event_type}_Zscore" traces in a single session (row)
        → one session_mean_trace (ignoring any traces that are all NaN,
        and trimming all to the same length).
        2) average those session_mean_traces across all sessions
        → one subject_mean_trace (again trimming to common length).
        Returns a DataFrame with columns:
        'subject_name', 'mean_trace', 'time_axis'
        """
        import numpy as np

        rows = []
        zcol = f"{event_type}_Zscore"
        tcol = f"{event_type}_Time_Axis"

        # group by subject
        for subj, subdf in df.groupby("subject_name"):
            session_means = []
            session_lengths = []

            # 1) build one mean‐trace per session
            for _, row in subdf.iterrows():
                raw_traces = row.get(zcol, []) or []
                clean = []
                for tr in raw_traces:
                    arr = np.asarray(tr, dtype=float)
                    if not np.all(np.isnan(arr)):
                        clean.append(arr)

                if not clean:
                    continue

                # trim all event‐traces in this session to the *shortest* length
                L = min(arr.shape[0] for arr in clean)
                clean = [arr[:L] for arr in clean]

                # now safely stack
                M = np.vstack(clean)                # shape (n_events, L)
                session_mean = np.nanmean(M, axis=0)
                session_means.append(session_mean)
                session_lengths.append(L)

            if not session_means:
                continue

            # 2) trim session‐means to the *shortest* session‐length
            Ls = [sm.shape[0] for sm in session_means]
            Lmin = min(Ls)
            trimmed = [sm[:Lmin] for sm in session_means]

            # stack across sessions
            S = np.vstack(trimmed)                 # shape (n_sessions, Lmin)
            subj_mean = np.nanmean(S, axis=0)

            # 3) pick a representative time‐axis, trimmed to Lmin
            #    Search through the very first row’s time axes:
            time_axis = None
            candidates = subdf.iloc[0].get(tcol, []) or []
            for cand in candidates:
                t_arr = np.asarray(cand, dtype=float)
                if t_arr.size >= Lmin:
                    time_axis = t_arr[:Lmin]
                    break
            if time_axis is None:
                # fallback
                time_axis = np.arange(Lmin)

            rows.append({
                "subject_name": subj,
                "mean_trace":   subj_mean,
                "time_axis":    time_axis
            })

        return pd.DataFrame(rows)






    def plot_group_mean_traces(self,
                           subj_traces: pd.DataFrame,
                           event_type: str,
                           brain_region: str,
                           color: str = None,
                           title: str = None,
                           ylim: tuple = None,
                           figsize=(8,5),
                           reward_line = True,
                           save_path: str = None):
        """
        Given the per-subject mean traces (output of compute_subject_mean_traces),
        filter by brain_region ('NAc' vs 'mPFC'), compute group mean ± SEM,
        and plot a single PSTH.
        """
        # filter by prefix n* vs p*
        if brain_region == "NAc":
            grp = subj_traces[subj_traces["subject_name"].str.startswith("n")]
            default_color = "#15616F"
        else:
            grp = subj_traces[subj_traces["subject_name"].str.startswith("p")]
            default_color = "#FFAF00"

        if grp.empty:
            print(f"No data for region {brain_region}")
            return

        # assemble array (n_subjects x n_time)
        M = np.vstack(grp["mean_trace"].values)
        mean_trace = np.nanmean(M, axis=0)
        sem_trace  = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])

        # time_axis (shared by all)
        t = grp.iloc[0]["time_axis"]

        # (skip downsampling here; use your downsample_data if desired)
        ds_time, ds_mean, ds_sem = t, mean_trace, sem_trace

        c = color or default_color

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ds_time, ds_mean, color=c, lw=3, label=f"{brain_region} Mean")
        ax.fill_between(ds_time,
                        ds_mean - ds_sem,
                        ds_mean + ds_sem,
                        color=c, alpha=0.4,
                        label="SEM")

        # vertical onset lines
        ax.axvline(0, color="k",      ls="--", lw=2)
        if reward_line:
            ax.axvline(4, color="#FF69B4",ls="--", lw=2)

        # labels & title
        ax.set_xlabel("Time (s)",         fontsize=14)
        ax.set_ylabel("z-scored ΔF/F",    fontsize=14)
        if title:
            ax.set_title(title, fontsize=18, fontweight='bold')
        else:
            ax.set_title(f"{brain_region} {event_type} PSTH", fontsize=18, fontweight='bold')

        # apply custom y-limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        # ticks & layout
        ax.set_xticks([-4, 0, 4, 10])
        ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)

        # remove top & right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()



    def old_plot_mean_psth(self,
                       winner_df: pd.DataFrame,
                       event_type: str,
                       brain_region: str,
                       bin_size: int = 100,
                       directory_path: str = None):
        """
        OLD METHOD: For each row in winner_df (every session), 
        collapse all per‐event traces → one session trace, then
        average across all sessions (regardless of subject).
        """
        import os, numpy as np, matplotlib.pyplot as plt

        # 1) pick only the region of interest
        prefix = 'n' if brain_region == 'NAc' else 'p'
        df_reg = winner_df[winner_df['subject_name'].str.startswith(prefix)]
        if df_reg.empty:
            print(f"No data for region {brain_region}")
            return

        # 2) collect one PSTH per session
        session_traces = []
        for _, row in df_reg.iterrows():
            # each row has a list of per‐event z‐score arrays
            evts = row[f'{event_type}_Zscore']
            if not isinstance(evts, (list, np.ndarray)) or len(evts) == 0:
                continue

            # stack [n_events x n_timebins]
            mat = np.vstack(evts)
            # mean across events → 1D array of length n_timebins
            session_trace = np.nanmean(mat, axis=0)
            session_traces.append(session_trace)

        if len(session_traces) == 0:
            print("No valid session traces found.")
            return

        # 3) stack all session traces → shape (n_sessions, n_timebins)
        M = np.vstack(session_traces)

        # 4) compute grand mean & SEM across sessions
        mean_psth = np.nanmean(M, axis=0)
        sem_psth  = np.nanstd(M,  axis=0) / np.sqrt(M.shape[0])

        # 5) grab time‐axis from the first row
        common_time_axis = df_reg.iloc[0][f'{event_type}_Time_Axis'][0]
        # downsample for smooth plotting
        def downsample(data, time, bs):
            n = len(data) // bs
            d = data[:n*bs].reshape(n, bs).mean(axis=1)
            t = time[:n*bs].reshape(n, bs).mean(axis=1)
            return d, t

        ds_mean, ts = downsample(mean_psth, common_time_axis, bin_size)
        ds_sem,  _  = downsample(sem_psth,   common_time_axis, bin_size)

        # 6) plot
        color = '#15616F' if brain_region=='NAc' else '#FFAF00'
        plt.figure(figsize=(8,5))
        plt.plot(ts, ds_mean, color=color, lw=3)
        plt.fill_between(ts, ds_mean-ds_sem, ds_mean+ds_sem, color=color, alpha=0.4)
        plt.axvline(0, color='k', ls='--', lw=2)
        plt.axvline(4, color='#FF69B4', ls='-', lw=2)
        plt.xlabel('Time from Cue (s)')
        plt.ylabel('Z-scored ΔF/F')
        plt.title(f'OLD MEAN PSTH: {event_type} ({brain_region})')
        plt.tight_layout()

        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"old_{brain_region}_{event_type}_meanPSTH.png"
            plt.savefig(os.path.join(directory_path, fname), dpi=300, bbox_inches='tight')

        plt.show()



    def plot_win_vs_loss(self,
                         metric_name: str,
                         behavior: str,
                         brain_region: str,
                         directory_path: str = None,
                         color_win: str = '#15616F',
                         color_loss: str = '#FFAF00',
                         figsize: tuple = (5,5),
                         pad_inches: float = 0.2):
        """
        Bar graph: Win vs Loss for a single [behavior metric_name].
        Automatically computes p-value & Cohen's d, places stats box on the right.
        """
        # 1) column key
        col = f"{behavior} {metric_name}"
        for df in (self.winner_df, self.loser_df):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from winner_df or loser_df")

        # 2) pick the right region
        prefix = 'n' if brain_region=='NAc' else 'p'
        win_df  = self.winner_df[self.winner_df['subject_name'].str.startswith(prefix)]
        lose_df = self.loser_df [self.loser_df ['subject_name'].str.startswith(prefix)]

        # 3) collapse each row’s list → single session number, then average across sessions per subject
        def subj_means(df):
            tmp = df[['subject_name', col]].copy()
            tmp[col] = tmp[col].apply(lambda lst:
                                      np.nanmean(lst) if isinstance(lst,(list,np.ndarray))
                                      else np.nan)
            tmp = tmp.dropna(subset=[col])
            # one value per subject
            return tmp.groupby('subject_name')[col].mean().values

        v_win  = subj_means(win_df)
        v_loss = subj_means(lose_df)

        # 4) stats
        tstat, pval = ttest_ind(v_win, v_loss, nan_policy='omit', equal_var=False)
        n1, n2 = len(v_win), len(v_loss)
        sd1, sd2 = np.nanstd(v_win, ddof=1), np.nanstd(v_loss, ddof=1)
        pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
        cohens_d = (np.nanmean(v_win) - np.nanmean(v_loss)) / pooled_sd

        # 5) means & sem
        m1, m2 = np.nanmean(v_win), np.nanmean(v_loss)
        sem1, sem2 = sd1/np.sqrt(n1), sd2/np.sqrt(n2)

        # 6) plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = [0,1]
        w = 0.6

        ax.bar(0, m1, width=w, yerr=sem1, capsize=8,
               color=color_win,  edgecolor='k', label='Win')
        ax.bar(1, m2, width=w, yerr=sem2, capsize=8,
               color=color_loss, edgecolor='k', label='Loss')

        # overlay each subject
        jitter = 0.05
        ax.scatter(np.zeros(n1)+np.random.uniform(-jitter, jitter, n1),
                   v_win,  facecolors='none', edgecolors='black',
                   s=80, alpha=0.8, zorder=3)
        ax.scatter(np.ones(n2)+np.random.uniform(-jitter, jitter, n2),
                   v_loss, facecolors='none', edgecolors='black',
                   s=80, alpha=0.8, zorder=3)

        # labels & title
        ax.set_xticks(x)
        ax.set_xticklabels(['Win','Loss'], fontsize=14)
        ax.set_xlabel('Outcome', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region} {behavior} {metric_name}: Win vs Loss",
                     fontsize=18, pad=12)

        # stats box
        stats_txt = f"p = {pval:.3f}\nCohen’s d = {cohens_d:.2f}"
        ax.text(1.05, 0.5, stats_txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray"))

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # 7) save
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_win_vs_loss.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)

        plt.show()
        return {'t_stat': tstat, 'p_value': pval, 'cohen_d': cohens_d}

    def plot_comp_vs_alone_by_subject(self,
                           df_alone,
                           df_comp,
                           metric_name: str,
                           behavior: str,
                           brain_region: str,
                           brain_color: str,
                           directory_path: str = None,
                           figsize: tuple = (5,5),
                           pad_inches: float = 0.2):
        """
        Compare Reward-Training (alone) vs Reward-Competition (win) by subject.
        Bars are the same color (brain_color), dots lie on the bar centers.
        """
        # 1) column name
        col = f"{behavior} {metric_name}"
        for df in (df_alone, df_comp):
            if col not in df.columns:
                raise KeyError(f"'{col}' missing from input dataframe")

        # 2) filter by brain region
        def pick(df):
            mask = df['subject_name'].str.startswith('n') if brain_region=='NAc' else df['subject_name'].str.startswith('p')
            return df[mask]

        a = pick(df_alone)
        c = pick(df_comp)

        # 3) flatten each row’s list to its mean, then average across sessions per subject
        def subj_means(df):
            tmp = df[['subject_name', col]].copy()
            tmp[col] = tmp[col].apply(lambda lst: np.nanmean(lst) 
                                    if isinstance(lst,(list,np.ndarray)) else np.nan)
            tmp = tmp.dropna(subset=[col])
            return tmp.groupby('subject_name')[col].mean()

        v1 = subj_means(a).values
        v2 = subj_means(c).values

        # 4) stats
        tstat, pval = ttest_ind(v1, v2, nan_policy='omit', equal_var=False)
        n1, n2 = len(v1), len(v2)
        sd1, sd2 = np.nanstd(v1,ddof=1), np.nanstd(v2,ddof=1)
        pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2)/(n1+n2-2))
        cohens_d = (np.nanmean(v1) - np.nanmean(v2)) / pooled_sd

        # 5) means & sem
        m1, m2 = np.nanmean(v1), np.nanmean(v2)
        sem1 = sd1/np.sqrt(n1)
        sem2 = sd2/np.sqrt(n2)

        # 6) plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = [0,1]
        width = 0.6

        # bars
        ax.bar(0, m1, width, yerr=sem1, capsize=8,
            color=brain_color, edgecolor='k', linewidth=2)
        ax.bar(1, m2, width, yerr=sem2, capsize=8,
            color=brain_color, edgecolor='k', linewidth=2)

        # dots on bar centers
        ax.scatter([0]*n1, v1, facecolors='white', edgecolors='black',
                s=120, linewidth=2, zorder=3)
        ax.scatter([1]*n2, v2, facecolors='white', edgecolors='black',
                s=120, linewidth=2, zorder=3)

        # labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Alone', 'Win (Comp)'], fontsize=14)
        ax.set_xlabel('Context', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region} {behavior} {metric_name}", fontsize=18, pad=12)

        # zero line
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        # stats box
        stats_txt = f"p = {pval:.3f}\nCohen’s d = {cohens_d:.2f}"
        ax.text(1.05, 0.5, stats_txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="gray"))

        # clean
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # save
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_alone_vs_comp.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)

        plt.show()
        return {'t_stat': tstat, 'p_value': pval, 'cohen_d': cohens_d}

    def plot_reward_comp_vs_alone(self,
                                df_alone,
                                df_comp,
                                metric_name: str,
                                behavior: str,
                                brain_region: str,
                                directory_path: str = None,
                                color_alone: str = '#15616F',
                                color_comp:  str = '#FFAF00',
                                figsize: tuple = (5,5),
                                pad_inches: float = 0.2):
        """
        A two-bar “Context” plot: Alone vs Competition for a single metric.
        Automatically computes p-value and Cohen’s d, places stats box on right.
        """
        # 1) column key
        col = f"{behavior} {metric_name}"
        if col not in df_alone.columns or col not in df_comp.columns:
            raise KeyError(f"Column {col} missing in one of the dataframes")

        # 2) pick out your region
        def sel_region(df):
            return df[df['subject_name'].str.startswith('n')] if brain_region=='NAc' \
                else df[df['subject_name'].str.startswith('p')]

        # 3) collapse lists → single session mean
        def collapse(df):
            arr = df[col].apply(lambda x: np.nanmean(x)
                                if isinstance(x,(list,np.ndarray)) else np.nan)
            return arr.dropna().values

        arr1 = collapse(sel_region(df_alone))
        arr2 = collapse(sel_region(df_comp))

        # 4) statistics
        tstat, pval = ttest_ind(arr1, arr2,
                                nan_policy='omit',
                                equal_var=False)
        n1, n2 = len(arr1), len(arr2)
        s1, s2 = np.nanstd(arr1,ddof=1), np.nanstd(arr2,ddof=1)
        # pooled SD for Cohen's d
        pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        cohend   = (np.nanmean(arr1)-np.nanmean(arr2)) / pooled_sd

        # 5) means & SEM
        m1, m2 = np.nanmean(arr1), np.nanmean(arr2)
        sem1    = s1/np.sqrt(n1)
        sem2    = s2/np.sqrt(n2)

        # 6) plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = np.array([0,1])
        w = 0.6

        # bars
        ax.bar(0, m1, yerr=sem1, width=w,
            color=color_alone, edgecolor='k', capsize=8,
            label='Alone')
        ax.bar(1, m2, yerr=sem2, width=w,
            color=color_comp,  edgecolor='k', capsize=8,
            label='Competition')

        # jittered points
        jitter = 0.08
        ax.scatter(np.zeros(n1)+np.random.uniform(-jitter,jitter,n1),
                arr1, facecolors='none', edgecolors='gray', s=80, alpha=0.7)
        ax.scatter(np.ones(n2) +np.random.uniform(-jitter,jitter,n2),
                arr2, facecolors='none', edgecolors='gray', s=80, alpha=0.7)

        # labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Alone','Competition'], fontsize=14)
        ax.set_xlabel('Context', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region} {behavior} {metric_name}", fontsize=18, pad=12)

        # stats textbox
        stats_txt = f"p = {pval:.3f}\nCohen’s d = {cohend:.2f}"
        ax.text(1.05, 0.5, stats_txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="gray"))

        # clean
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        plt.tight_layout()
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_alone_vs_comp.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)
        plt.show()

        # return stats if you want to log them
        return {'t_stat': tstat, 'p_value': pval, 'cohen_d': cohend}




    def plot_alone_win_loss_by_subject(self,
                                    df_alone: pd.DataFrame,
                                    df_win:   pd.DataFrame,
                                    df_loss:  pd.DataFrame,
                                    metric_name: str,
                                    behavior:    str,
                                    brain_region:str,
                                    brain_color: str,
                                    directory_path: str = None,
                                    figsize: tuple = (6,5),
                                    pad_inches: float = 0.2):
        """
        Compare three contexts by subject: Alone vs Win (first-tone) vs Loss (first-tone).

        Returns dict of statistics for the three pairwise comparisons.
        """
        col = f"{behavior} {metric_name}"
        # 1) sanity check
        for name, df in (("Alone",df_alone), ("Win",df_win), ("Loss",df_loss)):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from {name} DataFrame")

        # 2) pick region
        prefix = 'n' if brain_region=='NAc' else 'p'
        def _pick(df):
            return df[df['subject_name'].str.startswith(prefix)]
        a = _pick(df_alone)
        w = _pick(df_win)
        l = _pick(df_loss)

        # 3) collapse each row’s list → a single session mean → then mean across sessions per subject
        def subj_means(df):
            tmp = df[['subject_name', col]].copy()
            tmp[col] = tmp[col].apply(lambda lst:
                                    np.nanmean(lst) if isinstance(lst,(list,np.ndarray))
                                    else np.nan)
            tmp = tmp.dropna(subset=[col])
            return tmp.groupby('subject_name')[col].mean().values

        v1 = subj_means(a)
        v2 = subj_means(w)
        v3 = subj_means(l)

        # 4) pairwise statistics
        def pairwise(x,y):
            t, p = ttest_ind(x, y, nan_policy='omit', equal_var=False)
            n1,n2 = len(x),len(y)
            s1,s2 = np.nanstd(x,ddof=1),np.nanstd(y,ddof=1)
            # pooled SD
            sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
            d  = (np.nanmean(x)-np.nanmean(y))/sd if sd>0 else np.nan
            return t,p,d

        stats = {
            'alone_vs_win':  pairwise(v1,v2),
            'alone_vs_loss':pairwise(v1,v3),
            'win_vs_loss':  pairwise(v2,v3)
        }

        # 5) means & sem
        m1, m2, m3 = np.nanmean(v1), np.nanmean(v2), np.nanmean(v3)
        sem1 = np.nanstd(v1,ddof=1)/np.sqrt(len(v1))
        sem2 = np.nanstd(v2,ddof=1)/np.sqrt(len(v2))
        sem3 = np.nanstd(v3,ddof=1)/np.sqrt(len(v3))

        # 6) plot
        fig, ax = plt.subplots(figsize=figsize)
        x = [0,1,2]
        wbar = 0.6

        ax.bar(0, m1, width=wbar, yerr=sem1, capsize=6, color=brain_color, edgecolor='k', label='Alone')
        ax.bar(1, m2, width=wbar, yerr=sem2, capsize=6, color=brain_color, edgecolor='k', label='Win')
        ax.bar(2, m3, width=wbar, yerr=sem3, capsize=6, color=brain_color, edgecolor='k', label='Loss')

        # subject‐dots
        ax.scatter(np.zeros(len(v1))+0, v1, facecolors='white', edgecolors='black', s=80, zorder=3)
        ax.scatter(np.zeros(len(v2))+1, v2, facecolors='white', edgecolors='black', s=80, zorder=3)
        ax.scatter(np.zeros(len(v3))+2, v3, facecolors='white', edgecolors='black', s=80, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(['Alone','Win','Loss'], fontsize=14)
        ax.set_xlabel('Context', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region} {behavior} {metric_name}", fontsize=18, pad=12)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        # stats‐box
        txt = (
            f"A vs W: p={stats['alone_vs_win'][1]:.3f}, d={stats['alone_vs_win'][2]:.2f}\n"
            f"A vs L: p={stats['alone_vs_loss'][1]:.3f}, d={stats['alone_vs_loss'][2]:.2f}\n"
            f"W vs L: p={stats['win_vs_loss'][1]:.3f}, d={stats['win_vs_loss'][2]:.2f}"
        )
        ax.text(1.05, 0.5, txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_alone_win_loss.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)

        plt.show()
        return {
            'alone_vs_win':   {'t':stats['alone_vs_win'][0],'p':stats['alone_vs_win'][1],'d':stats['alone_vs_win'][2]},
            'alone_vs_loss': {'t':stats['alone_vs_loss'][0],'p':stats['alone_vs_loss'][1],'d':stats['alone_vs_loss'][2]},
            'win_vs_loss':   {'t':stats['win_vs_loss'][0],'p':stats['win_vs_loss'][1],'d':stats['win_vs_loss'][2]},
        }




    """*********************** First tones ******************************************"""
    def collapse_to_first_event(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For every column that starts with 'Tone ' or 'PE ',
        replace its list/array with a singleton list containing only the 0th element.
        """
        df = df.copy()
        # pick up every metric column for Tone and PE
        metric_cols = [c for c in df.columns 
                       if c.startswith("Tone ") or c.startswith("PE ")]
        for c in metric_cols:
            df[c] = df[c].apply(lambda lst:
                                [lst[0]] 
                                if isinstance(lst, (list, np.ndarray)) and len(lst)>0 
                                else [])
        return df

    def prep_rt_first_event(self, rt_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rewards‐Training: just collapse every per‐tone & per‐PE list to its first element.
        """
        return self.collapse_to_first_event(rt_df)

    def prep_rc_first_tone(self,
                       rc_df: pd.DataFrame,
                       outcome: str = 'win') -> pd.DataFrame:
        """
        Rewards-Competition:
        1) keep only rows where you [win|lose] on the *first* tone,
        2) then collapse every Tone*/PE* list to its 0th element.

        Parameters:
        rc_df    – the raw winner_df or loser_df
        outcome  – 'win' (default) to keep first‐tone winners,
                    'lose' to keep first‐tone losers
        """
        df = rc_df.copy()

        def _keep_row(row):
            arr = row.get('filtered_winner_array', [])
            # must have at least one entry
            if not isinstance(arr, list) or len(arr) == 0:
                return False
            first_is_win = (arr[0] == row['subject_name'])
            if outcome == 'win':
                return first_is_win
            elif outcome in ('lose', 'loss'):
                return not first_is_win
            else:
                raise ValueError("`outcome` must be either 'win' or 'lose'")
        
        mask = df.apply(_keep_row, axis=1)
        df = df[mask].reset_index(drop=True)

        # now collapse all of your per-event lists to their first element
        return self.collapse_to_first_event(df)



    def plot_metric_vs_count_per_session(self,
                                        df: pd.DataFrame,
                                        event_type: str = "Tone",
                                        metric: str = "AUC EI",
                                        brain_region: str = "NAc",
                                        condition: str = "Win",
                                        directory_path: str = None,
                                        figsize: tuple = (6,4),
                                        pad_inches: float = 0.1):
        """
        Scatter & best‐fit of session‐mean [event_type metric] vs number of
        {condition.lower()}s in that session, where count comes from
        len(filtered_winner_array) (ignoring any NaNs).
        """
        # 1) select region
        prefix = 'n' if brain_region == 'NAc' else 'p'
        sess = df[df['subject_name'].str.startswith(prefix)].copy()
        if sess.empty:
            print(f"No data for region {brain_region}")
            return

        col = f"{event_type} {metric}"
        records = []
        for _, row in sess.iterrows():
            # get your filtered_winner_array, count non‐NaN entries
            wins = row.get('filtered_winner_array', []) or []
            # remove any placeholders / NaNs
            n_events = sum(1 for w in wins if pd.notna(w))
            if n_events == 0:
                continue

            # compute the session‐mean of your metric list
            vals = row.get(col, []) or []
            # if vals is empty you might skip or set to NaN
            mean_val = float(np.nanmean(vals)) if len(vals)>0 else np.nan

            records.append((row['file name'], n_events, mean_val))

        sess_df = pd.DataFrame(records, columns=['file name','n_events','mean_metric'])
        if sess_df.empty:
            print("No valid sessions to plot.")
            return sess_df, np.nan, np.nan

        x = sess_df['n_events'].values
        y = sess_df['mean_metric'].values

        # 2) compute correlation + fit
        if len(x) > 1:
            r, p = pearsonr(x, y)
            m, b = np.polyfit(x, y, 1)
        else:
            r = p = m = b = np.nan

        # 3) plot
        plt.figure(figsize=figsize)
        plt.scatter(x, y,
                    facecolors='none', edgecolors='black', s=80, linewidth=2)
        if not np.isnan(m):
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, m*xs + b, 'r--', lw=2, label=f"fit: y={m:.2f}x+{b:.2f}")
            plt.legend()

        plt.xlabel(f"Number of {condition.lower()}s in session", fontsize=14)
        plt.ylabel(f"Mean {event_type} {metric}", fontsize=14)
        plt.title(
            f"{brain_region} {event_type} {metric} vs {condition}/session\n"
            f"r={r:.2f}, p={p:.3f}",
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout(pad=pad_inches)

        # 4) save if requested
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{event_type}_{metric.replace(' ','')}_vs_{condition.lower()}s.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)

        plt.show()
        return sess_df, r, p






    def rc_plot_event_scatter_by_outcome(self,
                                        column: str,
                                        condition: str = 'win',
                                        brain_region: str = 'NAc',
                                        color: str = 'C0',
                                        individual_dots: bool = False,
                                        xlabel: str = None,
                                        ylabel: str = None,
                                        title: str = None,
                                        yrange: tuple = None,
                                        min_count: int = None):
        """
        Like `plot_event_scatter_by_outcome`, but only include sessions where
        the number of events (wins or losses) >= min_count.
        """
        # 1) pick the correct dataframe
        if condition.lower().startswith('win'):
            df0 = self.winner_df.copy()
        elif condition.lower().startswith('loss'):
            df0 = self.loser_df.copy()
        else:
            raise ValueError("condition must be 'win' or 'loss'")

        # 2) filter by brain region prefix
        prefix = 'n' if brain_region=='NAc' else 'p'
        df0 = df0[df0['subject_name'].str.startswith(prefix)]

        # 3) if requested, only keep sessions with >= min_count events
        if min_count is not None:
            df0 = df0[df0[column].apply(lambda lst: isinstance(lst,(list,np.ndarray)) and len(lst) >= min_count)]
            if df0.empty:
                print(f"No sessions with ≥{min_count} {condition}s in {brain_region}")
                return

        # 4) gather all the per-session arrays
        arrays = [np.asarray(v) for v in df0[column]
                if isinstance(v,(list,np.ndarray)) and len(v)>0]
        max_len = max(arr.shape[0] for arr in arrays)
        means   = np.zeros(max_len)
        sems    = np.zeros(max_len)
        idxs    = np.arange(1, max_len+1)

        fig, ax = plt.subplots(figsize=(8,5))

        for i in range(max_len):
            ys = [arr[i] for arr in arrays if arr.shape[0]>i and not np.isnan(arr[i])]
            if individual_dots:
                ax.scatter([i+1]*len(ys), ys,
                        facecolors='none', edgecolors=color, alpha=0.6, s=60)
            means[i] = np.mean(ys) if ys else np.nan
            sems[i]  = (np.std(ys,ddof=1)/np.sqrt(len(ys))) if len(ys)>1 else 0

        # 5) mean ± SEM
        ax.errorbar(idxs, means, yerr=sems,
                    fmt='o', lw=2, capsize=5, color=color,
                    label='Mean ± SEM')

        # 6) pearson & fit
        valid = ~np.isnan(means)
        r,p = pearsonr(idxs[valid], means[valid])
        m,b,_,_,_ = linregress(idxs[valid], means[valid])
        ax.plot(idxs, m*idxs+b, '--', color=color, lw=2,
                label=f'fit: y={m:.2f}x+{b:.2f}')

        # 7) labels & title
        noun = "Wins" if condition.lower().startswith('win') else "Losses"
        ax.set_xlabel(xlabel or f"{noun} index", fontsize=14)
        ax.set_ylabel(ylabel or column, fontsize=14)
        ax.set_title(title or
                    f"{brain_region} {column} vs. {noun} (≥{min_count})\n"
                    f"r={r:.2f}, p={p:.3f}",
                    fontsize=16)
        ax.set_xticks(idxs)
        if yrange: ax.set_ylim(yrange)

        # clean up
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
        plt.tight_layout()
        plt.show()

        return {'r': r, 'p': p, 'slope': m, 'intercept': b}



    def rc_plot_event_scatter_grid_by_outcome(self,
            column:        str,
            condition:     str   = 'win',
            brain_region:  str   = 'NAc',
            color:         str   = 'C0',
            individual_dots: bool = False,
            xlabel:        str   = None,
            ylabel:        str   = None,
            title:         str   = None,
            yrange:        tuple = None,    # if you want to override per-panel
            min_count:     int   = None,
            ncols:         int   = 5,
            figsize_per:   tuple = (2,2),
            max_events:    int   = 19):
        """
        Plot each session’s per-event values as a small scatter + best-fit line,
        arranged in a grid.  X runs 1→max_events; each panel gets its own Y-span.
        """
        # 1) pick the right dataframe
        if condition.lower().startswith('win'):
            df0 = self.winner_df.copy()
        elif condition.lower().startswith('loss'):
            df0 = self.loser_df.copy()
        else:
            raise ValueError("condition must be 'win' or 'loss'")

        # 2) region filter
        prefix = 'n' if brain_region=='NAc' else 'p'
        df0 = df0[df0['subject_name'].str.startswith(prefix)]

        # 3) optional minimum count
        if min_count is not None:
            df0 = df0[df0[column].apply(
                lambda lst: isinstance(lst,(list,np.ndarray)) and len(lst)>=min_count)]
            if df0.empty:
                print(f"No sessions with ≥{min_count} {condition}s in {brain_region}")
                return

        # 4) gather
        records = []
        for _, row in df0.iterrows():
            raw = row.get(column, [])
            if isinstance(raw,(list,np.ndarray)) and len(raw)>0:
                arr = np.asarray(raw, dtype=float)  # None→nan
                records.append((row['file name'], arr))

        if not records:
            print(f"No valid sessions for {column}")
            return

        # 5) make grid
        N     = len(records)
        nrows = math.ceil(N / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(figsize_per[0]*ncols,
                                        figsize_per[1]*nrows),
                                sharex=False, sharey=False)
        axes = axes.flatten()

        # 6) plot each
        for i, (fname, arr) in enumerate(records):
            ax = axes[i]

            # pad/truncate to max_events
            pad = np.full(max_events, np.nan)
            L   = min(len(arr), max_events)
            pad[:L] = arr[:L]
            xs = np.arange(1, max_events+1)
            ys = pad

            # valid mask
            valid = ~np.isnan(ys)

            # scatter
            if individual_dots:
                ax.scatter(xs[valid], ys[valid],
                        facecolors='none', edgecolors=color,
                        s=40, alpha=0.6)
            else:
                ax.scatter(xs[valid], ys[valid],
                        facecolors='none', edgecolors=color,
                        s=40)

            # best-fit
            if valid.sum() > 1:
                m,b,_,_,_ = linregress(xs[valid], ys[valid])
                ax.plot([1, max_events],
                        [m*1 + b, m*max_events + b],
                        '--', color=color, lw=1)

            # now set a tight y-lim around data
            if yrange is None and valid.sum()>0:
                y_min, y_max = ys[valid].min(), ys[valid].max()
                pad_amt = 0.1 * (y_max - y_min) if (y_max > y_min) else 0.5
                ax.set_ylim(y_min - pad_amt, y_max + pad_amt)
            elif yrange is not None:
                ax.set_ylim(yrange)

            ax.set_xlim(1, max_events)
            ax.set_title(fname, fontsize=8)
            ax.set_xticks([1,5,10,15, max_events])
            ax.tick_params(labelsize=6)

        # 7) blank any extras
        for j in range(len(records), len(axes)):
            axes[j].axis('off')

        # 8) super-labels & suptitle
        noun = "Wins" if condition.lower().startswith('win') else "Losses"
        plt.suptitle(title or f"{brain_region} {column} per-session ({condition})", fontsize=12)
        fig.text(0.5, 0.04,
                xlabel or f"Event index (1→{max_events})",
                ha='center', fontsize=10)
        fig.text(0.04, 0.5,
                ylabel or column,
                va='center', rotation='vertical', fontsize=10)
        plt.tight_layout(rect=[0.05,0.05,1,0.95])
        plt.show()




    def plot_slope_vs_count(self,
                            event_type: str,
                            metric_name: str,
                            brain_region: str,
                            condition: str = 'win',
                            directory_path: str = None,
                            figsize: tuple = (6,4),
                            pad_inches: float = 0.1):
        """
        For each session where the subject [wins|loses], fit a line to
        [event index → metric] and extract its slope; then scatter
        slope vs. number of wins (or losses).
        
        Parameters
        ----------
        event_type   : "Tone" or "PE"
        metric_name  : e.g. "AUC", "Max Peak", etc.
        brain_region : "NAc" or "mPFC"
        condition    : "win" or "loss"
        """
        # 1) pick the right df
        if condition.lower().startswith('win'):
            df0 = self.winner_df.copy()
        elif condition.lower().startswith('loss'):
            df0 = self.loser_df.copy()
        else:
            raise ValueError("condition must be 'win' or 'loss'")
        
        # 2) filter by region prefix
        prefix = 'n' if brain_region=='NAc' else 'p'
        df0 = df0[df0['subject_name'].str.startswith(prefix)]
        if df0.empty:
            print(f"No {condition}s in region {brain_region}")
            return
        
        # 3) gather per-session slopes & counts
        col = f"{event_type} {metric_name}"
        records = []
        for _, row in df0.iterrows():
            vals = row.get(col, [])
            if not isinstance(vals, (list, np.ndarray)) or len(vals) < 2:
                continue
            arr = np.asarray(vals, dtype=float)
            idxs = np.arange(1, arr.size+1)
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                continue
            
            # fit event-index → metric
            slope, intercept, _, _, _ = linregress(idxs[mask], arr[mask])
            n_events = mask.sum()
            records.append((row['file name'], n_events, slope))
        
        if not records:
            print("No valid sessions to plot.")
            return
        
        sess_df = (
            pd.DataFrame(records, columns=['file name','n_events','slope'])
        )
        x = sess_df['n_events'].values
        y = sess_df['slope'].values

        # 4) stats & fit
        r, p = pearsonr(x, y) if len(x)>1 else (np.nan, np.nan)
        m, b = np.polyfit(x, y, 1)  if len(x)>1 else (np.nan, np.nan)

        # 5) plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, y, facecolors='none', edgecolors='black', s=80, label=None)
        if not np.isnan(m):
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, m*xs + b, 'r--', lw=2, label=f"fit: y={m:.2f}x+{b:.2f}")
        ax.set_xlabel(f"Number of {condition.capitalize()}s in session", fontsize=14)
        ax.set_ylabel(f"Slope of {event_type} {metric_name}",    fontsize=14)
        ax.set_title(
            f"{brain_region} {event_type} {metric_name} slope vs {condition.capitalize()}\n"
            f"r = {r:.2f}, p = {p:.3f}",
            fontsize=16, fontweight='bold'
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='gray', lw=1, ls='--')
        plt.tight_layout(pad=pad_inches)
        
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{event_type}_{metric_name.replace(' ','')}_slope_vs_{condition}.png"
            fig.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)
        plt.show()
        
        return {'r':r, 'p':p, 'slope_vs_count_fit':(m,b)}


    def compute_event_relative_responses(self,
                                        event_type: str    = "Tone",
                                        metric:    str    = "AUC",
                                        df:        pd.DataFrame = None
                                    ) -> pd.DataFrame:
        """
        For each session in df (defaults to self.da_df), take the list in column
        f"{event_type} {metric}"
        and compute its values relative to the 1st entry.

        Returns a DataFrame with columns
        'file name'   : session identifier
        'relative'    : np.ndarray of length n_events with values A_i/A_1
        'percent_drop': np.ndarray of length n_events with (1 - A_i/A_1) * 100
        """
        if df is None:
            df = self.da_df

        col = f"{event_type} {metric}"
        records = []
        for _, row in df.iterrows():
            vals = row.get(col, []) or []
            arr  = np.array(vals, dtype=float)
            if arr.size < 1:
                continue
            first = arr[0]
            if np.isnan(first) or first == 0:
                continue

            rel  = arr / first
            drop = (1.0 - rel) * 100.0

            records.append({
                'file name':     row['file name'],
                'relative':      rel,
                'percent_drop':  drop
            })

        return pd.DataFrame(records)


    def plot_event_decay(self,
                        event_type:   str     = "Tone",
                        metric:       str     = "AUC",
                        brain_region: str     = "NAc",
                        use_percent:  bool    = False,
                        max_events:   int     = 19,
                        figsize:      tuple   = (6,4),
                        color:        str     = None,
                        xlabel:       str     = None,
                        ylabel:       str     = None,
                        title:        str     = None,
                        save_path:    str     = None):
        """
        Plot how {event_type} {metric} decays over successive events,
        normalized to the 1st event.  Shows mean ± SEM of relative (or % drop).
        """
        # 1) get relative responses
        rel_df = self.compute_event_relative_responses(
                    event_type=event_type, metric=metric)

        # 2) filter brain region
        prefix = 'n' if brain_region=="NAc" else 'p'
        rel_df = rel_df[rel_df['file name'].str.startswith(prefix)]
        if rel_df.empty:
            print(f"No data for region {brain_region}")
            return

        # 3) build session × event matrix, padding with NaN
        mats = []
        key = 'percent_drop' if use_percent else 'relative'
        for arr in rel_df[key]:
            v = np.array(arr, dtype=float)
            if v.size >= 1:
                v = v[:max_events]
                if v.size < max_events:
                    v = np.concatenate([v,
                                        np.full(max_events - v.size, np.nan)])
                mats.append(v)
        M = np.vstack(mats)  # shape (n_sessions, max_events)

        # 4) mean & SEM across sessions
        mean = np.nanmean(M, axis=0)
        sem  = np.nanstd(M,  axis=0, ddof=1) / np.sqrt(M.shape[0])

        # 5) plot
        events = np.arange(1, max_events+1)
        c      = color or ("#15616F" if brain_region=="NAc" else "#FFAF00")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(events, mean,    color=c, lw=3)
        ax.fill_between(events, mean-sem, mean+sem, color=c, alpha=0.3)

        ax.set_xlim(1,   max_events)
        ax.set_xticks(np.arange(1, max_events+1, 2))
        ax.set_xlabel(xlabel or f"{event_type} index", fontsize=14)

        ylab_def = ("% drop from 1st"
                if use_percent
                else "relative to 1st")
        ax.set_ylabel(ylabel or f"{metric} ({ylab_def})", fontsize=14)

        if title:
            ax.set_title(title, fontsize=16)
        else:
            desc = "% drop" if use_percent else "rel to 1st"
            ax.set_title(f"{brain_region} {event_type} {metric} decay ({desc})",
                        fontsize=16)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def plot_alone_win_loss_by_subject_firstN(self,
        df_alone: pd.DataFrame,
        df_win:   pd.DataFrame,
        df_loss:  pd.DataFrame,
        metric_name: str,
        behavior:    str,
        brain_region:str,
        brain_color: str,
        directory_path: str = None,
        figsize: tuple = (6,5),
        pad_inches: float = 0.2,
        max_events: int = 19):
        """
        Compare three contexts by subject: Alone vs Win vs Loss, 
        but only counting the first `max_events` events in Win/Loss.
        """
        col = f"{behavior} {metric_name}"
        # 1) sanity check
        for name, df in (("Alone",df_alone), ("Win",df_win), ("Loss",df_loss)):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from {name} DataFrame")

        # 2) pick region
        prefix = 'n' if brain_region.lower()=='nac' else 'p'
        def _pick(df):
            return df[df['subject_name'].str.startswith(prefix)].copy()
        a = _pick(df_alone)
        w = _pick(df_win)
        l = _pick(df_loss)

        # 3) truncate each session’s list to the first max_events
        for df in (w, l):
            df[col] = df[col].apply(
                lambda lst: lst[:max_events] 
                            if isinstance(lst, (list, np.ndarray))
                            else []
            )

        # 4) collapse each row’s list → single session mean → then mean across sessions per subject
        def subj_means(df):
            tmp = df[['subject_name', col]].copy()
            tmp[col] = tmp[col].apply(lambda lst:
                                np.nanmean(lst) if isinstance(lst,(list,np.ndarray)) and len(lst)>0
                                else np.nan)
            tmp = tmp.dropna(subset=[col])
            return tmp.groupby('subject_name')[col].mean().values

        v1 = subj_means(a)
        v2 = subj_means(w)
        v3 = subj_means(l)

        # 5) pairwise stats
        def pairwise(x,y):
            t, p = ttest_ind(x, y, nan_policy='omit', equal_var=False)
            n1,n2 = len(x), len(y)
            s1,s2 = np.nanstd(x,ddof=1), np.nanstd(y,ddof=1)
            sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            d  = (np.nanmean(x)-np.nanmean(y)) / sd if sd>0 else np.nan
            return t,p,d

        stats = {
            'alone_vs_win':   pairwise(v1,v2),
            'alone_vs_loss':  pairwise(v1,v3),
            'win_vs_loss':    pairwise(v2,v3)
        }

        # 6) means & sem
        m1,m2,m3 = np.nanmean(v1), np.nanmean(v2), np.nanmean(v3)
        sem1 = np.nanstd(v1,ddof=1)/np.sqrt(len(v1))
        sem2 = np.nanstd(v2,ddof=1)/np.sqrt(len(v2))
        sem3 = np.nanstd(v3,ddof=1)/np.sqrt(len(v3))

        # 7) plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = [0,1,2]; wbar = 0.6
        ax.bar(0, m1, width=wbar, yerr=sem1, capsize=6,
            color=brain_color, edgecolor='k', label='Alone')
        ax.bar(1, m2, width=wbar, yerr=sem2, capsize=6,
            color=brain_color, edgecolor='k', label='Win')
        ax.bar(2, m3, width=wbar, yerr=sem3, capsize=6,
            color=brain_color, edgecolor='k', label='Loss')

        # subject‐dots
        ax.scatter(np.zeros(len(v1))+0, v1, facecolors='white', edgecolors='black', s=80, zorder=3)
        ax.scatter(np.zeros(len(v2))+1, v2, facecolors='white', edgecolors='black', s=80, zorder=3)
        ax.scatter(np.zeros(len(v3))+2, v3, facecolors='white', edgecolors='black', s=80, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(['Alone','Win','Loss'], fontsize=14)
        ax.set_xlabel('Context', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region.upper()} {behavior} {metric_name}\n(first {max_events} wins/losses)", 
                    fontsize=18, pad=12)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        # stats‐box
        txt = (
            f"A vs W: p={stats['alone_vs_win'][1]:.3f}, d={stats['alone_vs_win'][2]:.2f}\n"
            f"A vs L: p={stats['alone_vs_loss'][1]:.3f}, d={stats['alone_vs_loss'][2]:.2f}\n"
            f"W vs L: p={stats['win_vs_loss'][1]:.3f}, d={stats['win_vs_loss'][2]:.2f}"
        )
        ax.text(1.05, 0.5, txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_first{max_events}_alone_win_loss.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)

        plt.show()
        return {
            'alone_vs_win':   {'t':stats['alone_vs_win'][0],'p':stats['alone_vs_win'][1],'d':stats['alone_vs_win'][2]},
            'alone_vs_loss': {'t':stats['alone_vs_loss'][0],'p':stats['alone_vs_loss'][1],'d':stats['alone_vs_loss'][2]},
            'win_vs_loss':   {'t':stats['win_vs_loss'][0],'p':stats['win_vs_loss'][1],'d':stats['win_vs_loss'][2]},
        }


    def build_transition_dfs(self,
                            transitions=("win-win", "win-loss", "loss-win", "loss-loss"),
                            drop_ties=True,
                            drop_tangles=True):
        """
        Creates per-transition dataframes that mimic winner_df/loser_df structure:
        each metric column becomes a list filtered down to events belonging to that transition.

        Output saved as:
        self.transition_dfs[transition] = df

        Requires:
        self.da_df contains 'filtered_winner_array' and the *_Zscore/*_Time_Axis list columns.
        """

        df = self.da_df.copy()

        id_cols = ["subject_name", "file name", "trial"]
        metric_cols = [c for c in df.columns if c not in id_cols]

        def _as_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x if isinstance(x, list) else []

        def _label_outcome(w, subject):
            # tie
            if pd.isna(w):
                return "tie"
            # tangle placeholder
            if isinstance(w, str) and w.strip().lower() == "tangle":
                return "tangle"
            # normal
            return "win" if w == subject else "loss"

        def _transition_for_index(outcomes, i):
            """
            Your rule:
            - if current is tangle -> 'tangle'
            - if previous is tangle -> 'tangle-win' or 'tangle-loss' (do not look further back)
            - otherwise prev-current for win/loss only
            """
            curr = outcomes[i]

            if curr == "tangle":
                return "tangle"

            # treat tie as non-transition unless you want them
            if curr == "tie":
                return "tie"

            if i == 0:
                return f"start-{curr}"

            prev = outcomes[i-1]
            if prev == "tangle":
                return f"tangle-{curr}"
            if prev == "tie":
                return f"tie-{curr}"

            return f"{prev}-{curr}"

        def _filter_lists_by_mask(seq, keep_mask):
            seq = _as_list(seq)
            out = []
            for i, val in enumerate(seq):
                if i < len(keep_mask) and keep_mask[i]:
                    out.append(val)
            return out

        transition_dfs = {}

        for tr in transitions:
            tr = tr.lower()
            # build per-row masks, then apply to all metric columns
            kept_rows = []
            for _, row in df.iterrows():
                wins = _as_list(row.get("filtered_winner_array", []))
                subj = row["subject_name"]

                outcomes = [_label_outcome(w, subj) for w in wins]
                trans = [_transition_for_index(outcomes, i) for i in range(len(outcomes))]

                # event keep mask for this transition
                keep_mask = []
                for t in trans:
                    if drop_tangles and ("tangle" in t):
                        keep_mask.append(False)
                        continue
                    if drop_ties and ("tie" in t):
                        keep_mask.append(False)
                        continue
                    keep_mask.append(t == tr)

                # apply mask to every metric column
                new_row = {k: row[k] for k in id_cols}
                for col in metric_cols:
                    new_row[col] = _filter_lists_by_mask(row[col], keep_mask)

                # keep only sessions that actually have >=1 event
                # choose a representative list column to check; Tone_Zscore usually exists
                rep = new_row.get("Tone_Zscore", [])
                if isinstance(rep, list) and len(rep) > 0:
                    kept_rows.append(new_row)

            transition_dfs[tr] = pd.DataFrame(kept_rows)

        self.transition_dfs = transition_dfs
        return transition_dfs
        
    


    def plot_transition_psth(
        self,
        transition: str,
        event_type: str,                 # "Tone" or "PE"
        brain_region: str,
        time_window_s: tuple = None,     # example: (0 - 4, 10)
        ylim: tuple = None,
        title: str = None,
        reward_line = True
    ):
        """
        event_type sets what is time zero, because the underlying traces are already aligned to that event.
        time_window_s crops the plotted window.
        """
        def _crop_subject_traces(subj_traces: pd.DataFrame, time_window_s: tuple) -> pd.DataFrame:
            """
            Expects columns: subject_name, mean_trace, time_axis
            Crops each trace to the requested time window.
            """
            if time_window_s is None:
                return subj_traces

            t0, t1 = time_window_s
            out_rows = []

            for _, r in subj_traces.iterrows():
                t = np.asarray(r["time_axis"], dtype=float)
                y = np.asarray(r["mean_trace"], dtype=float)

                L = min(len(t), len(y))
                t = t[:L]
                y = y[:L]

                m = (t >= t0) & (t <= t1)
                if not np.any(m):
                    continue

                out_rows.append({
                    "subject_name": r["subject_name"],
                    "time_axis": t[m],
                    "mean_trace": y[m],
                })

            return pd.DataFrame(out_rows)
        
        if not hasattr(self, "transition_dfs"):
            raise RuntimeError("Run build_transition_dfs() first.")

        tr = transition.lower()
        if tr not in self.transition_dfs:
            raise KeyError(f"Transition '{transition}' not found.")

        df_tr = self.transition_dfs[tr]

        subj_traces = self.compute_subject_mean_traces(
            df_tr,
            event_type=event_type
        )

        subj_traces = _crop_subject_traces(subj_traces, time_window_s)

        return self.plot_group_mean_traces(
            subj_traces=subj_traces,
            event_type=event_type,
            brain_region=brain_region,
            ylim=ylim,
            title=title or f"{event_type} PSTH | {brain_region} | {tr}",
            reward_line = reward_line
        )


