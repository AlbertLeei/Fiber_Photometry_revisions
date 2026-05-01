import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


FEATURE_COLUMNS = [
    "distance_head_res__head_int",
    "angle_head_res__head_int_deg",
    "distance_head_res__hind_int",
    "angle_head_res__hind_int_deg",
    "distance_head_int__hind_res",
    "angle_head_int__hind_res_deg",
    "distance_hind_res__hind_int",
    "velocity_resident",
    "velocity_intruder",
]

COLUMN_RENAME_MAP = {
    "head-head_dist": "distance_head_res__head_int",
    "res_head-int_head_angle": "angle_head_res__head_int_deg",
    "res_head-int_hind_dist": "distance_head_res__hind_int",
    "res_head-int_hind_angle": "angle_head_res__hind_int_deg",
    "int_head-res_hind_dist": "distance_head_int__hind_res",
    "int_head-res_hind_angle": "angle_head_int__hind_res_deg",
    "hind-hind_dist": "distance_hind_res__hind_int",
    "resident_velocity": "velocity_resident",
    "intruder_velocity": "velocity_intruder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated pose-only Gaussian HMM for social state discovery."
    )
    parser.add_argument(
        "--input",
        default="Pose_Tracking/home_cage_pose_DA_all.csv",
        help="Input CSV. Defaults to Pose_Tracking/home_cage_pose_DA_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="Pose_Tracking/hmm_social_states_gpu_output",
        help="Directory where CSV summaries and figures will be written.",
    )
    # parser.add_argument("--device", default="cuda", help="Torch device. Default: cuda")
    parser.add_argument("--device", default="cpu", help="Torch device. Default: cpu")
    parser.add_argument("--brain-region", nargs="*", default=None, help="Optional brain region filter.")
    parser.add_argument("--mouse", nargs="*", default=None, help="Optional mouse filter.")
    parser.add_argument("--intruder", nargs="*", default=None, help="Optional intruder filter.")
    parser.add_argument("--gap-threshold-s", type=float, default=0.75, help="Gap threshold for sequence resets.")
    parser.add_argument("--min-sequence-len", type=int, default=150, help="Minimum sequence length after resets.")
    parser.add_argument("--min-states", type=int, default=3, help="Minimum state count to evaluate.")
    parser.add_argument("--max-states", type=int, default=8, help="Maximum state count to evaluate.")
    parser.add_argument("--n-init", type=int, default=3, help="Number of random initializations per state count.")
    parser.add_argument("--max-iter", type=int, default=40, help="Maximum EM iterations.")
    parser.add_argument("--kmeans-iters", type=int, default=15, help="K-means iterations used for initialization.")
    parser.add_argument("--random-state", type=int, default=7, help="Base random seed.")
    parser.add_argument("--pre-s", type=float, default=3.0, help="Seconds before a transition for DA alignment.")
    parser.add_argument("--post-s", type=float, default=5.0, help="Seconds after a transition for DA alignment.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. "
            "Install a CUDA-enabled PyTorch build and confirm `torch.cuda.is_available()` is True."
        )
    return torch.device(device_name)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], float]:
    df = pd.read_csv(args.input).rename(columns=COLUMN_RENAME_MAP)

    for col in FEATURE_COLUMNS + ["time_s", "zscore_DA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "behavior_active" not in df.columns:
        df["behavior_active"] = np.nan

    required_cols = ["time_s", "mouse_identity", "brain_region", "intruder_identity"] + FEATURE_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.brain_region:
        df = df[df["brain_region"].isin(args.brain_region)].copy()
    if args.mouse:
        df = df[df["mouse_identity"].isin(args.mouse)].copy()
    if args.intruder:
        df = df[df["intruder_identity"].isin(args.intruder)].copy()

    df = df.sort_values(["mouse_identity", "brain_region", "intruder_identity", "time_s"]).reset_index(drop=True)
    df["behavior_label"] = df["behavior_active"].fillna("").replace("", "None")
    df["base_block"] = (
        df["mouse_identity"].astype(str)
        + "|"
        + df["brain_region"].astype(str)
        + "|"
        + df["intruder_identity"].astype(str)
    )

    gap_breaks = df.groupby("base_block")["time_s"].diff().fillna(0).gt(args.gap_threshold_s)
    df["segment_in_block"] = gap_breaks.groupby(df["base_block"]).cumsum().astype(int)
    df["sequence_id"] = df["base_block"] + "|seg" + df["segment_in_block"].astype(str)

    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    seq_lengths = df.groupby("sequence_id").size()
    valid_sequences = seq_lengths[seq_lengths >= args.min_sequence_len].index
    df = df[df["sequence_id"].isin(valid_sequences)].copy()

    z_df_parts = []
    for _, seq_df in df.groupby("sequence_id", sort=False):
        seq = seq_df.copy()
        values = seq[FEATURE_COLUMNS].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0, keepdims=True)
        std = np.nanstd(values, axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        for i, col in enumerate(FEATURE_COLUMNS):
            seq[f"{col}_z"] = (values[:, i] - mean[0, i]) / std[0, i]
        z_df_parts.append(seq)

    z_df = pd.concat(z_df_parts, axis=0).reset_index(drop=True)
    z_features = [f"{col}_z" for col in FEATURE_COLUMNS]

    dt = float(z_df.groupby("sequence_id")["time_s"].diff().median())
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.1

    return z_df, z_features, dt


class TorchDiagonalGaussianHMM:
    def __init__(
        self,
        n_states: int,
        n_iter: int = 30,
        tol: float = 1e-3,
        min_covar: float = 1e-3,
        random_state: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        kmeans_iters: int = 15,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.kmeans_iters = kmeans_iters

    def _split(self, X: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
        chunks = []
        start = 0
        for length in lengths:
            chunks.append(X[start : start + length])
            start += length
        return chunks

    def _torch_kmeans(self, X: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.random_state)

        indices = torch.randperm(X.shape[0], generator=generator, device=self.device)[: self.n_states]
        centers = X[indices].clone()

        for _ in range(self.kmeans_iters):
            dist2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
            labels = torch.argmin(dist2, dim=1)

            new_centers = []
            for state in range(self.n_states):
                mask = labels == state
                if torch.any(mask):
                    new_centers.append(X[mask].mean(dim=0))
                else:
                    fallback_idx = torch.randint(
                        0,
                        X.shape[0],
                        (1,),
                        generator=generator,
                        device=self.device,
                    )[0]
                    new_centers.append(X[fallback_idx])
            new_centers = torch.stack(new_centers, dim=0)
            if torch.allclose(new_centers, centers, atol=1e-4, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        return centers

    def _initialize(self, X: torch.Tensor) -> None:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.random_state)

        self.means_ = self._torch_kmeans(X)
        dist2 = ((X[:, None, :] - self.means_[None, :, :]) ** 2).sum(dim=2)
        labels = torch.argmin(dist2, dim=1)

        covars = []
        for state in range(self.n_states):
            members = X[labels == state]
            if members.shape[0] == 0:
                fallback_idx = torch.randint(0, X.shape[0], (32,), generator=generator, device=self.device)
                members = X[fallback_idx]
            var = members.var(dim=0, unbiased=False).clamp_min(self.min_covar)
            covars.append(var)
        self.covars_ = torch.stack(covars, dim=0)

        self.startprob_ = torch.full((self.n_states,), 1.0 / self.n_states, dtype=self.dtype, device=self.device)
        trans = torch.full(
            (self.n_states, self.n_states),
            1.0 / self.n_states,
            dtype=self.dtype,
            device=self.device,
        )
        trans = trans + torch.eye(self.n_states, dtype=self.dtype, device=self.device) * 2.0
        self.transmat_ = trans / trans.sum(dim=1, keepdim=True)

    def _log_emission(self, X: torch.Tensor) -> torch.Tensor:
        diff = X[:, None, :] - self.means_[None, :, :]
        log_det = torch.log(self.covars_).sum(dim=1)
        maha = ((diff**2) / self.covars_[None, :, :]).sum(dim=2)
        return -0.5 * (X.shape[1] * math.log(2 * math.pi) + log_det[None, :] + maha)

    def _forward_backward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_B = self._log_emission(X)
        log_start = torch.log(self.startprob_.clamp_min(1e-12))
        log_trans = torch.log(self.transmat_.clamp_min(1e-12))

        T = X.shape[0]
        alpha = torch.empty((T, self.n_states), dtype=self.dtype, device=self.device)
        alpha[0] = log_start + log_B[0]
        for t in range(1, T):
            alpha[t] = log_B[t] + torch.logsumexp(alpha[t - 1][:, None] + log_trans, dim=0)

        beta = torch.zeros((T, self.n_states), dtype=self.dtype, device=self.device)
        for t in range(T - 2, -1, -1):
            beta[t] = torch.logsumexp(log_trans + log_B[t + 1][None, :] + beta[t + 1][None, :], dim=1)

        loglik = torch.logsumexp(alpha[-1], dim=0)
        log_gamma = alpha + beta - loglik
        gamma = torch.exp(log_gamma)

        xi_sum = torch.zeros((self.n_states, self.n_states), dtype=self.dtype, device=self.device)
        for t in range(T - 1):
            log_xi_t = (
                alpha[t][:, None]
                + log_trans
                + log_B[t + 1][None, :]
                + beta[t + 1][None, :]
                - loglik
            )
            xi_sum += torch.exp(log_xi_t)
        return loglik, gamma, xi_sum

    def fit(self, X: torch.Tensor, lengths: list[int]) -> "TorchDiagonalGaussianHMM":
        self._initialize(X)
        sequences = self._split(X, lengths)
        prev_loglik = None
        self.history_ = []

        for _ in range(self.n_iter):
            total_loglik = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            start_counts = torch.zeros((self.n_states,), dtype=self.dtype, device=self.device)
            trans_counts = torch.zeros((self.n_states, self.n_states), dtype=self.dtype, device=self.device)
            gamma_sum = torch.zeros((self.n_states,), dtype=self.dtype, device=self.device)
            gamma_obs = torch.zeros((self.n_states, X.shape[1]), dtype=self.dtype, device=self.device)
            gamma_obs2 = torch.zeros((self.n_states, X.shape[1]), dtype=self.dtype, device=self.device)

            for seq in sequences:
                loglik, gamma, xi_sum = self._forward_backward(seq)
                total_loglik += loglik
                start_counts += gamma[0]
                trans_counts += xi_sum
                gamma_sum += gamma.sum(dim=0)
                gamma_obs += gamma.transpose(0, 1) @ seq
                gamma_obs2 += gamma.transpose(0, 1) @ (seq**2)

            self.startprob_ = start_counts.clamp_min(1e-12)
            self.startprob_ = self.startprob_ / self.startprob_.sum()

            trans_counts = trans_counts + 1e-6
            self.transmat_ = trans_counts / trans_counts.sum(dim=1, keepdim=True)

            gamma_sum_safe = gamma_sum[:, None].clamp_min(1e-12)
            self.means_ = gamma_obs / gamma_sum_safe
            second_moment = gamma_obs2 / gamma_sum_safe
            self.covars_ = (second_moment - self.means_**2).clamp_min(self.min_covar)

            total_loglik_value = float(total_loglik.item())
            self.history_.append(total_loglik_value)
            if prev_loglik is not None and abs(total_loglik_value - prev_loglik) < self.tol:
                break
            prev_loglik = total_loglik_value

        self.loglik_ = self.score(X, lengths)
        return self

    def score(self, X: torch.Tensor, lengths: list[int]) -> float:
        total = 0.0
        for seq in self._split(X, lengths):
            total += float(self._forward_backward(seq)[0].item())
        return total

    def bic(self, X: torch.Tensor, lengths: list[int]) -> float:
        n_features = X.shape[1]
        n_params = (self.n_states - 1) + self.n_states * (self.n_states - 1) + 2 * self.n_states * n_features
        return -2 * self.score(X, lengths) + n_params * math.log(X.shape[0])

    def predict(self, X: torch.Tensor, lengths: list[int]) -> np.ndarray:
        paths = []
        log_start = torch.log(self.startprob_.clamp_min(1e-12))
        log_trans = torch.log(self.transmat_.clamp_min(1e-12))
        for seq in self._split(X, lengths):
            log_B = self._log_emission(seq)
            T = seq.shape[0]
            delta = torch.empty((T, self.n_states), dtype=self.dtype, device=self.device)
            psi = torch.empty((T, self.n_states), dtype=torch.long, device=self.device)
            delta[0] = log_start + log_B[0]
            psi[0] = 0
            for t in range(1, T):
                scores = delta[t - 1][:, None] + log_trans
                psi[t] = torch.argmax(scores, dim=0)
                delta[t] = log_B[t] + torch.max(scores, dim=0).values

            states = torch.empty((T,), dtype=torch.long, device=self.device)
            states[-1] = torch.argmax(delta[-1])
            for t in range(T - 2, -1, -1):
                states[t] = psi[t + 1, states[t + 1]]
            paths.append(states.detach().cpu().numpy())
        return np.concatenate(paths)


def fit_hmm_grid(
    X: torch.Tensor,
    lengths: list[int],
    min_states: int,
    max_states: int,
    n_init: int,
    max_iter: int,
    kmeans_iters: int,
    random_state: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[int, TorchDiagonalGaussianHMM]]:
    rows = []
    models = {}

    for n_states in range(min_states, max_states + 1):
        best_model = None
        best_loglik = -np.inf
        for init_idx in range(n_init):
            seed = random_state + 100 * n_states + init_idx
            model = TorchDiagonalGaussianHMM(
                n_states=n_states,
                n_iter=max_iter,
                random_state=seed,
                device=device,
                kmeans_iters=kmeans_iters,
            ).fit(X, lengths)

            if model.loglik_ > best_loglik:
                best_loglik = model.loglik_
                best_model = model

        assert best_model is not None
        models[n_states] = best_model
        rows.append(
            {
                "n_states": n_states,
                "loglik": best_model.loglik_,
                "bic": best_model.bic(X, lengths),
                "iterations": len(best_model.history_),
            }
        )
    return pd.DataFrame(rows).sort_values("n_states"), models


def build_dwell_table(frame: pd.DataFrame, dt: float) -> pd.DataFrame:
    rows = []
    for seq_id, seq_df in frame.groupby("sequence_id", sort=False):
        states = seq_df["hmm_state"].to_numpy()
        times = seq_df["time_s"].to_numpy()
        start = 0
        for idx in range(1, len(seq_df) + 1):
            changed = idx == len(seq_df) or states[idx] != states[start]
            if changed:
                rows.append(
                    {
                        "sequence_id": seq_id,
                        "hmm_state": int(states[start]),
                        "n_frames": idx - start,
                        "duration_s": float(times[idx - 1] - times[start] + dt),
                    }
                )
                start = idx
    return pd.DataFrame(rows)


def extract_transition_events(
    frame: pd.DataFrame,
    source_state: int | None,
    target_state: int | None,
    pre_frames: int,
    post_frames: int,
) -> pd.DataFrame:
    rows = []
    for seq_id, seq_df in frame.groupby("sequence_id", sort=False):
        seq_df = seq_df.reset_index(drop=True)
        states = seq_df["hmm_state"].to_numpy()
        da = seq_df["zscore_DA"].to_numpy(dtype=float)
        time_s = seq_df["time_s"].to_numpy(dtype=float)
        for idx in range(1, len(seq_df)):
            prev_state = int(states[idx - 1])
            curr_state = int(states[idx])
            if prev_state == curr_state:
                continue
            if source_state is not None and prev_state != source_state:
                continue
            if target_state is not None and curr_state != target_state:
                continue

            start = idx - pre_frames
            stop = idx + post_frames + 1
            if start < 0 or stop > len(seq_df):
                continue

            trace = da[start:stop]
            rel_time = time_s[start:stop] - time_s[idx]
            post_mask = rel_time >= 0
            peak = np.nanmax(trace[post_mask])
            peak_latency = rel_time[post_mask][np.nanargmax(trace[post_mask])]
            auc = np.trapz(trace[post_mask], rel_time[post_mask])
            pre_mask = rel_time < 0
            pre_slope = np.polyfit(rel_time[pre_mask], trace[pre_mask], 1)[0] if pre_mask.sum() >= 2 else np.nan

            rows.append(
                {
                    "sequence_id": seq_id,
                    "entry_index": idx,
                    "source_state": prev_state,
                    "target_state": curr_state,
                    "transition_label": f"{prev_state}->{curr_state}",
                    "entry_time_s": time_s[idx],
                    "peak": float(peak),
                    "peak_latency_s": float(peak_latency),
                    "auc_post": float(auc),
                    "pre_slope": float(pre_slope),
                }
            )
    return pd.DataFrame(rows)


def save_model_selection_plot(model_selection: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(model_selection["n_states"], model_selection["loglik"], marker="o")
    axes[0].set_title("Model Log Likelihood")
    axes[0].set_xlabel("Number of states")
    axes[0].set_ylabel("Log likelihood")

    axes[1].plot(model_selection["n_states"], model_selection["bic"], marker="o", color="tab:red")
    axes[1].set_title("Model BIC")
    axes[1].set_xlabel("Number of states")
    axes[1].set_ylabel("BIC")
    plt.tight_layout()
    fig.savefig(output_dir / "model_selection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_state_heatmap(state_feature_summary: pd.DataFrame, output_dir: Path) -> None:
    heatmap_data = state_feature_summary[FEATURE_COLUMNS]
    fig, ax = plt.subplots(figsize=(12, 4 + 0.35 * len(heatmap_data)))
    im = ax.imshow(heatmap_data.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(heatmap_data)))
    ax.set_yticklabels([f"State {s}" for s in heatmap_data.index])
    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=60, ha="right")
    ax.set_title("Mean Raw Pose Features by HMM State")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    fig.savefig(output_dir / "state_feature_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    z_df, z_features, dt = load_and_prepare_data(args)
    X_np = z_df[z_features].to_numpy(dtype=np.float32)
    lengths = z_df.groupby("sequence_id", sort=False).size().tolist()
    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)

    model_selection, models = fit_hmm_grid(
        X=X,
        lengths=lengths,
        min_states=args.min_states,
        max_states=args.max_states,
        n_init=args.n_init,
        max_iter=args.max_iter,
        kmeans_iters=args.kmeans_iters,
        random_state=args.random_state,
        device=device,
    )

    best_k = int(model_selection.sort_values("bic").iloc[0]["n_states"])
    best_model = models[best_k]
    print(f"Selected BEST_K = {best_k}")

    z_df["hmm_state"] = best_model.predict(X, lengths)

    state_feature_summary = z_df.groupby("hmm_state")[FEATURE_COLUMNS + ["zscore_DA"]].mean()
    state_counts = z_df["hmm_state"].value_counts().sort_index().rename("n_frames")
    dwell_df = build_dwell_table(z_df, dt)
    state_dwell = dwell_df.groupby("hmm_state")[["n_frames", "duration_s"]].median().rename(
        columns={"n_frames": "median_dwell_frames", "duration_s": "median_dwell_s"}
    )
    state_overview = pd.concat([state_counts, state_dwell], axis=1)

    transition_counts = np.zeros((best_k, best_k), dtype=int)
    for _, seq_df in z_df.groupby("sequence_id", sort=False):
        states = seq_df["hmm_state"].to_numpy()
        for a, b in zip(states[:-1], states[1:]):
            transition_counts[a, b] += 1
    transition_probs = transition_counts / np.clip(transition_counts.sum(axis=1, keepdims=True), 1, None)
    transition_probs_df = pd.DataFrame(transition_probs)

    state_by_label = pd.crosstab(z_df["hmm_state"], z_df["behavior_label"])
    state_by_label_frac = state_by_label.div(state_by_label.sum(axis=1), axis=0)
    label_by_state_frac = state_by_label.div(state_by_label.sum(axis=0), axis=1)

    distance_cols = [
        "distance_head_res__head_int",
        "distance_head_res__hind_int",
        "distance_head_int__hind_res",
        "distance_hind_res__hind_int",
    ]
    velocity_cols = ["velocity_resident", "velocity_intruder"]

    mean_dist_by_state = state_feature_summary[distance_cols].mean(axis=1)
    baseline_state = int(mean_dist_by_state.idxmax())
    face_like_state = int(state_feature_summary["distance_head_res__head_int"].idxmin())
    rear_like_state = int(state_feature_summary["distance_head_res__hind_int"].idxmin())
    high_motion_state = int(state_feature_summary[velocity_cols].mean(axis=1).idxmax())
    state_suggestions = pd.DataFrame(
        {
            "suggested_state": [
                baseline_state,
                face_like_state,
                rear_like_state,
                high_motion_state,
            ]
        },
        index=[
            "baseline_state_suggestion",
            "face_like_state_suggestion",
            "rear_like_state_suggestion",
            "high_motion_state_suggestion",
        ],
    )

    pre_frames = int(round(args.pre_s / dt))
    post_frames = int(round(args.post_s / dt))
    transition_specs = [
        (baseline_state, face_like_state, "baseline_to_face_like"),
        (baseline_state, rear_like_state, "baseline_to_rear_like"),
        (face_like_state, rear_like_state, "face_like_to_rear_like"),
        (rear_like_state, baseline_state, "rear_like_to_baseline"),
    ]

    transition_summaries = []
    for source_state, target_state, label in transition_specs:
        events = extract_transition_events(z_df, source_state, target_state, pre_frames, post_frames)
        if events.empty:
            continue
        summary = events[["peak", "peak_latency_s", "auc_post", "pre_slope"]].agg(["mean", "median", "count"]).T
        summary["transition"] = label
        transition_summaries.append(summary.reset_index().rename(columns={"index": "metric"}))
        events.to_csv(output_dir / f"{label}_events.csv", index=False)

    transition_summary_df = (
        pd.concat(transition_summaries, ignore_index=True) if transition_summaries else pd.DataFrame()
    )

    model_selection.to_csv(output_dir / "model_selection.csv", index=False)
    z_df.to_csv(output_dir / "state_assignments.csv", index=False)
    state_feature_summary.to_csv(output_dir / "state_feature_summary.csv")
    state_overview.to_csv(output_dir / "state_overview.csv")
    transition_probs_df.to_csv(output_dir / "transition_probabilities.csv", index=False)
    state_by_label.to_csv(output_dir / "state_by_label_counts.csv")
    state_by_label_frac.to_csv(output_dir / "state_by_label_fraction_of_state.csv")
    label_by_state_frac.to_csv(output_dir / "label_by_state_fraction_of_label.csv")
    state_suggestions.to_csv(output_dir / "state_suggestions.csv")
    if not transition_summary_df.empty:
        transition_summary_df.to_csv(output_dir / "transition_summary.csv", index=False)

    save_model_selection_plot(model_selection, output_dir)
    save_state_heatmap(state_feature_summary, output_dir)

    print("Run complete.")
    print(f"Rows analyzed: {len(z_df)}")
    print(f"Sequences analyzed: {z_df['sequence_id'].nunique()}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
