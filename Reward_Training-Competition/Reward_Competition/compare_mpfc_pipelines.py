import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from trial_class import Trial


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    description: str
    apply_lowpass: bool = True
    apply_pre_irls_highpass: bool = False
    pre_irls_highpass_preserve_mean: bool = True
    apply_post_dff_highpass: bool = False
    apply_double_exp: bool = False
    regression_method: str = "irls"


PIPELINES = [
    PipelineConfig(
        name="current_default",
        description="Low-pass -> high-pass on channels -> IRLS -> dF/F",
        apply_lowpass=True,
        apply_pre_irls_highpass=True,
        pre_irls_highpass_preserve_mean=True,
        regression_method="irls",
    ),
    PipelineConfig(
        name="current_default_ols",
        description="Low-pass -> high-pass on channels -> OLS -> dF/F",
        apply_lowpass=True,
        apply_pre_irls_highpass=True,
        pre_irls_highpass_preserve_mean=True,
        regression_method="ols",
    ),
    PipelineConfig(
        name="highpass_centered_pre_irls",
        description="Low-pass -> high-pass on channels without mean restoration -> IRLS -> dF/F",
        apply_lowpass=True,
        apply_pre_irls_highpass=True,
        pre_irls_highpass_preserve_mean=False,
        regression_method="irls",
    ),
    PipelineConfig(
        name="no_highpass",
        description="Low-pass -> IRLS -> dF/F",
        apply_lowpass=True,
        regression_method="irls",
    ),
    PipelineConfig(
        name="no_highpass_ols",
        description="Low-pass -> OLS -> dF/F",
        apply_lowpass=True,
        regression_method="ols",
    ),
    PipelineConfig(
        name="highpass_after_dff",
        description="Low-pass -> IRLS -> dF/F -> high-pass on dF/F",
        apply_lowpass=True,
        apply_post_dff_highpass=True,
        regression_method="irls",
    ),
    PipelineConfig(
        name="double_exp_detrend",
        description="Low-pass -> double-exponential detrend -> IRLS -> dF/F",
        apply_lowpass=True,
        apply_double_exp=True,
        regression_method="irls",
    ),
    PipelineConfig(
        name="double_exp_detrend_ols",
        description="Low-pass -> double-exponential detrend -> OLS -> dF/F",
        apply_lowpass=True,
        apply_double_exp=True,
        regression_method="ols",
    ),
    PipelineConfig(
        name="raw_irls_only",
        description="No low-pass, no high-pass -> IRLS -> dF/F",
        apply_lowpass=False,
        regression_method="irls",
    ),
    PipelineConfig(
        name="raw_ols_only",
        description="No low-pass, no high-pass -> OLS -> dF/F",
        apply_lowpass=False,
        regression_method="ols",
    ),
]


IRLS_ANALYSIS_CONFIG = PipelineConfig(
    name="irls_constant_sweep",
    description="Low-pass -> high-pass on channels -> IRLS -> dF/F",
    apply_lowpass=True,
    apply_pre_irls_highpass=True,
    pre_irls_highpass_preserve_mean=True,
    regression_method="irls",
)


def mad(arr: np.ndarray) -> float:
    median = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - median)))


def bandpower_fraction(signal: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if signal.size < 8:
        return np.nan

    freqs, power = welch(signal, fs=fs, nperseg=min(2048, signal.size))
    total_power = np.trapz(power, freqs)
    if total_power <= 0:
        return np.nan

    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.nan

    band_power = np.trapz(power[mask], freqs[mask])
    return float(band_power / total_power)


def transient_snr(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if signal.size < 8:
        return np.nan

    noise = mad(np.diff(signal))
    if noise == 0:
        return np.nan
    return float(np.nanstd(signal) / noise)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_pipeline(
    trial_path: str,
    da_stream: str,
    isos_stream: str,
    config: PipelineConfig,
    target_fs: float,
    trim_start: float,
    trim_end: float,
    lowpass_cutoff: float,
    highpass_cutoff: float,
    irls_constant: float,
    zscore_method: str,
) -> dict:
    trial = Trial(trial_path, da_stream, isos_stream)

    if target_fs < trial.fs:
        trial.downsample(target_fs=target_fs)
    if trim_start > 0:
        trial.remove_initial_LED_artifact(t=trim_start)
    if trim_end > 0:
        trial.remove_final_data_segment(t=trim_end)

    raw_da = trial.updated_DA.copy()
    raw_isos = trial.updated_ISOS.copy()

    if config.apply_lowpass:
        trial.lowpass_filter(cutoff_hz=lowpass_cutoff)
    if config.apply_double_exp:
        trial.basline_drift_double_exponential()
    elif config.apply_pre_irls_highpass:
        trial.baseline_drift_highpass(
            cutoff=highpass_cutoff,
            preserve_mean=config.pre_irls_highpass_preserve_mean,
        )

    pre_irls_da = trial.updated_DA.copy()
    pre_irls_isos = trial.updated_ISOS.copy()

    if config.regression_method == "irls":
        trial.motion_correction_align_channels_IRLS(IRLS_constant=irls_constant)
    elif config.regression_method == "ols":
        trial.motion_correction_align_channels_OLS()
    else:
        raise ValueError(f"Unsupported regression_method: {config.regression_method}")

    trial.compute_dFF()

    if config.apply_post_dff_highpass:
        trial.highpass_baseline_drift_dFF(cutoff=highpass_cutoff)

    trial.compute_zscore(method=zscore_method)

    residual = trial.updated_DA - trial.isosbestic_fitted

    result = {
        "config": config,
        "trial": trial,
        "raw_da": raw_da,
        "raw_isos": raw_isos,
        "pre_irls_da": pre_irls_da,
        "pre_irls_isos": pre_irls_isos,
        "residual": residual,
        "metrics": {
            "pipeline": config.name,
            "description": config.description,
            "regression_method": config.regression_method,
            "da_fit_corr": safe_corr(trial.updated_DA, trial.isosbestic_fitted),
            "residual_mad": mad(residual),
            "residual_mean_abs": float(np.nanmean(np.abs(residual))),
            "dff_std": float(np.nanstd(trial.dFF)),
            "zscore_std": float(np.nanstd(trial.zscore)),
            "low_freq_fraction_lt_0p01Hz": bandpower_fraction(trial.dFF, trial.fs, (0.0, 0.01)),
            "signal_fraction_0p01_to_0p5Hz": bandpower_fraction(trial.dFF, trial.fs, (0.01, 0.5)),
            "transient_snr": transient_snr(trial.dFF),
        },
    }
    return result


def rank_pipelines(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    ranked["fit_rank"] = ranked["da_fit_corr"].rank(ascending=False, method="min")
    ranked["residual_rank"] = ranked["residual_mad"].rank(ascending=True, method="min")
    ranked["drift_rank"] = ranked["low_freq_fraction_lt_0p01Hz"].rank(ascending=True, method="min")
    ranked["snr_rank"] = ranked["transient_snr"].rank(ascending=False, method="min")
    ranked["composite_rank_score"] = (
        ranked["fit_rank"] + ranked["residual_rank"] + ranked["drift_rank"] + ranked["snr_rank"]
    )
    ranked = ranked.sort_values(["composite_rank_score", "residual_mad", "low_freq_fraction_lt_0p01Hz"])
    return ranked


def rank_irls_constants(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    ranked["fit_rank"] = ranked["da_fit_corr"].rank(ascending=False, method="min")
    ranked["residual_rank"] = ranked["residual_mad"].rank(ascending=True, method="min")
    ranked["drift_rank"] = ranked["low_freq_fraction_lt_0p01Hz"].rank(ascending=True, method="min")
    ranked["snr_rank"] = ranked["transient_snr"].rank(ascending=False, method="min")
    ranked["composite_rank_score"] = (
        ranked["fit_rank"] + ranked["residual_rank"] + ranked["drift_rank"] + ranked["snr_rank"]
    )
    ranked = ranked.sort_values(
        ["composite_rank_score", "residual_mad", "low_freq_fraction_lt_0p01Hz", "irls_constant"]
    )
    return ranked


def _window_mask(timestamps: np.ndarray, start_time: float | None, end_time: float | None) -> np.ndarray:
    mask = np.ones_like(timestamps, dtype=bool)
    if start_time is not None:
        mask &= timestamps >= start_time
    if end_time is not None:
        mask &= timestamps <= end_time
    return mask


def plot_pipeline_comparison(
    results: list[dict],
    start_time: float | None = None,
    end_time: float | None = None,
    output_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(len(results), 3, figsize=(18, 4 * len(results)), sharex=False)
    if len(results) == 1:
        axes = np.array([axes])

    for row_idx, result in enumerate(results):
        trial = result["trial"]
        cfg = result["config"]
        mask = _window_mask(trial.timestamps, start_time, end_time)
        t = trial.timestamps[mask]

        ax0, ax1, ax2 = axes[row_idx]

        ax0.plot(t, result["pre_irls_da"][mask], color="steelblue", linewidth=1.2, label="DA")
        ax0.plot(t, result["pre_irls_isos"][mask], color="darkorange", linewidth=1.2, label="ISOS")
        ax0.set_title(f"{cfg.name}: pre-IRLS traces")
        ax0.set_ylabel("Voltage (V)")
        ax0.grid(alpha=0.3)
        if row_idx == 0:
            ax0.legend(loc="upper right")

        ax1.plot(t, trial.updated_DA[mask], color="steelblue", linewidth=1.2, label="DA")
        ax1.plot(t, trial.isosbestic_fitted[mask], color="darkorange", linewidth=1.2, label="Fitted ISOS")
        ax1.set_title(f"{cfg.name}: IRLS fit")
        ax1.grid(alpha=0.3)
        if row_idx == 0:
            ax1.legend(loc="upper right")

        ax2.plot(t, trial.dFF[mask], color="green", linewidth=1.2, label="dF/F")
        ax2.plot(t, trial.zscore[mask], color="purple", linewidth=1.0, alpha=0.8, label="z-score")
        ax2.set_title(f"{cfg.name}: normalized output")
        ax2.grid(alpha=0.3)
        if row_idx == 0:
            ax2.legend(loc="upper right")

        score = result["metrics"]
        subtitle = (
            f"corr={score['da_fit_corr']:.3f}, resid_MAD={score['residual_mad']:.4f}, "
            f"low-freq={score['low_freq_fraction_lt_0p01Hz']:.3f}, snr={score['transient_snr']:.3f}"
        )
        ax2.set_xlabel("Time (s)")
        ax2.text(1.01, 0.5, subtitle, transform=ax2.transAxes, va="center", fontsize=9)

    plt.tight_layout()
    if output_path:
        ensure_parent_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def run_irls_constant_analysis(
    trial_path: str,
    da_stream: str,
    isos_stream: str,
    irls_constants: list[float],
    target_fs: float,
    trim_start: float,
    trim_end: float,
    lowpass_cutoff: float,
    highpass_cutoff: float,
    zscore_method: str,
) -> list[dict]:
    results = []
    for irls_constant in irls_constants:
        print(
            f"Running IRLS constant {irls_constant:g} with "
            f"{IRLS_ANALYSIS_CONFIG.description}"
        )
        result = run_pipeline(
            trial_path=trial_path,
            da_stream=da_stream,
            isos_stream=isos_stream,
            config=IRLS_ANALYSIS_CONFIG,
            target_fs=target_fs,
            trim_start=trim_start,
            trim_end=trim_end,
            lowpass_cutoff=lowpass_cutoff,
            highpass_cutoff=highpass_cutoff,
            irls_constant=irls_constant,
            zscore_method=zscore_method,
        )
        result["metrics"]["irls_constant"] = float(irls_constant)
        results.append(result)
    return results


def plot_irls_constant_comparison(
    results: list[dict],
    start_time: float | None = None,
    end_time: float | None = None,
    output_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(len(results), 3, figsize=(18, 4 * len(results)), sharex=False)
    if len(results) == 1:
        axes = np.array([axes])

    for row_idx, result in enumerate(results):
        trial = result["trial"]
        irls_constant = result["metrics"]["irls_constant"]
        mask = _window_mask(trial.timestamps, start_time, end_time)
        t = trial.timestamps[mask]

        ax0, ax1, ax2 = axes[row_idx]

        ax0.plot(t, result["pre_irls_da"][mask], color="steelblue", linewidth=1.2, label="DA")
        ax0.plot(t, result["pre_irls_isos"][mask], color="darkorange", linewidth=1.2, label="ISOS")
        ax0.set_title(f"IRLS c={irls_constant:g}: pre-IRLS traces")
        ax0.set_ylabel("Voltage (V)")
        ax0.grid(alpha=0.3)
        if row_idx == 0:
            ax0.legend(loc="upper right")

        ax1.plot(t, trial.updated_DA[mask], color="steelblue", linewidth=1.2, label="DA")
        ax1.plot(t, trial.isosbestic_fitted[mask], color="darkorange", linewidth=1.2, label="Fitted ISOS")
        ax1.set_title(f"IRLS c={irls_constant:g}: fit")
        ax1.grid(alpha=0.3)
        if row_idx == 0:
            ax1.legend(loc="upper right")

        ax2.plot(t, trial.dFF[mask], color="green", linewidth=1.2, label="dF/F")
        ax2.plot(t, trial.zscore[mask], color="purple", linewidth=1.0, alpha=0.8, label="z-score")
        ax2.set_title(f"IRLS c={irls_constant:g}: normalized output")
        ax2.grid(alpha=0.3)
        if row_idx == 0:
            ax2.legend(loc="upper right")

        score = result["metrics"]
        subtitle = (
            f"corr={score['da_fit_corr']:.3f}, resid_MAD={score['residual_mad']:.4f}, "
            f"low-freq={score['low_freq_fraction_lt_0p01Hz']:.3f}, snr={score['transient_snr']:.3f}"
        )
        ax2.set_xlabel("Time (s)")
        ax2.text(1.01, 0.5, subtitle, transform=ax2.transAxes, va="center", fontsize=9)

    plt.tight_layout()
    if output_path:
        ensure_parent_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare alternative mPFC preprocessing pipelines on a single TDT trial."
    )
    parser.add_argument("trial_path", help="Path to a single TDT block folder.")
    parser.add_argument("--da-stream", default="_465A", help="DA stream key. Default: _465A")
    parser.add_argument("--isos-stream", default="_405A", help="ISOS stream key. Default: _405A")
    parser.add_argument("--target-fs", type=float, default=100.0, help="Downsample target in Hz.")
    parser.add_argument("--trim-start", type=float, default=150.0, help="Seconds to remove from the start.")
    parser.add_argument("--trim-end", type=float, default=10.0, help="Seconds to remove from the end.")
    parser.add_argument("--lowpass-cutoff", type=float, default=3.0, help="Low-pass cutoff in Hz.")
    parser.add_argument("--highpass-cutoff", type=float, default=0.001, help="High-pass cutoff in Hz.")
    parser.add_argument("--irls-constant", type=float, default=3.0, help="Tukey biweight constant for IRLS.")
    parser.add_argument(
        "--irls-constants",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional separate IRLS-constant sweep, for example: "
            "--irls-constants 1.5 2 3 4 5"
        ),
    )
    parser.add_argument(
        "--zscore-method",
        default="standard",
        choices=["standard", "baseline", "modified"],
        help="Z-score method passed to Trial.compute_zscore().",
    )
    parser.add_argument("--start-time", type=float, default=None, help="Optional plot window start in seconds.")
    parser.add_argument("--end-time", type=float, default=None, help="Optional plot window end in seconds.")
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional CSV path for the ranked metrics table.",
    )
    parser.add_argument(
        "--figure-out",
        default=None,
        help="Optional output path for the comparison figure.",
    )
    parser.add_argument(
        "--irls-metrics-out",
        default=None,
        help="Optional CSV path for the ranked IRLS-constant summary table.",
    )
    parser.add_argument(
        "--irls-figure-out",
        default=None,
        help="Optional output path for the IRLS-constant comparison figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = []
    for config in PIPELINES:
        print(f"Running {config.name}: {config.description}")
        result = run_pipeline(
            trial_path=args.trial_path,
            da_stream=args.da_stream,
            isos_stream=args.isos_stream,
            config=config,
            target_fs=args.target_fs,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            lowpass_cutoff=args.lowpass_cutoff,
            highpass_cutoff=args.highpass_cutoff,
            irls_constant=args.irls_constant,
            zscore_method=args.zscore_method,
        )
        results.append(result)

    metrics_df = pd.DataFrame([result["metrics"] for result in results])
    ranked_df = rank_pipelines(metrics_df)

    pd.set_option("display.max_columns", None)
    print("\nRanked pipeline summary:")
    print(ranked_df.to_string(index=False))
    print(
        "\nHeuristic readout: lower residual/drift is usually better, while higher fit correlation "
        "and transient SNR usually mean the pipeline preserves more usable dynamics."
    )

    if args.metrics_out:
        ensure_parent_dir(args.metrics_out)
        ranked_df.to_csv(args.metrics_out, index=False)

    plot_pipeline_comparison(
        results,
        start_time=args.start_time,
        end_time=args.end_time,
        output_path=args.figure_out,
    )

    if args.irls_constants:
        irls_results = run_irls_constant_analysis(
            trial_path=args.trial_path,
            da_stream=args.da_stream,
            isos_stream=args.isos_stream,
            irls_constants=args.irls_constants,
            target_fs=args.target_fs,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            lowpass_cutoff=args.lowpass_cutoff,
            highpass_cutoff=args.highpass_cutoff,
            zscore_method=args.zscore_method,
        )

        irls_metrics_df = pd.DataFrame([result["metrics"] for result in irls_results])
        irls_ranked_df = rank_irls_constants(irls_metrics_df)

        print("\nRanked IRLS constant summary:")
        print(irls_ranked_df.to_string(index=False))
        print(
            "\nHeuristic readout: smaller constants downweight outliers more aggressively, "
            "while larger constants behave more like ordinary least squares."
        )

        if args.irls_metrics_out:
            ensure_parent_dir(args.irls_metrics_out)
            irls_ranked_df.to_csv(args.irls_metrics_out, index=False)

        plot_irls_constant_comparison(
            irls_results,
            start_time=args.start_time,
            end_time=args.end_time,
            output_path=args.irls_figure_out,
        )


if __name__ == "__main__":
    main()
