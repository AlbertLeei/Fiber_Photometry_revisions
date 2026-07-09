"""
Shared figure sizing, styling, subplot layout, and export helpers.

Use this file from any experiment script or notebook:

    from figure_settings import apply_plot_style, create_figure_grid, save_figure
    apply_plot_style()

For aligned multi-panel figures:

    fig, axes = create_figure_grid(1, 2, sharex="row")
    axes[0, 0].plot(x, y_e)
    axes[0, 1].plot(x, y_f)
    save_figure(fig, "Figure_1_EF", formats=("png", "svg", "pdf"))

SVG and PDF files open cleanly in Adobe Illustrator. Text remains editable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FigureStyle:
    """Central defaults for every experiment figure."""

    panel_width: float = 3.2
    panel_height: float = 2.4
    full_width: float = 6.8
    row_gap: float = 0.45
    col_gap: float = 0.45
    dpi: int = 300
    font_family: str = "Arial"
    font_size: float = 8
    title_size: float = 9
    label_size: float = 8
    tick_size: float = 7
    legend_size: float = 7
    panel_label_size: float = 11
    line_width: float = 1.5
    axis_line_width: float = 0.8
    tick_width: float = 0.8
    tick_length: float = 3.0
    marker_size: float = 4.0
    transparent: bool = True
    save_formats: tuple[str, ...] = ("png", "svg", "pdf")


DEFAULT_STYLE = FigureStyle()

_CURRENT_STYLE = DEFAULT_STYLE
_ORIGINAL_PLT_FIGURE = plt.figure
_ORIGINAL_PLT_SUBPLOTS = plt.subplots
_SIZE_PATCH_ENABLED = False


def apply_plot_style(enforce_size: bool = False, respect_explicit_figsize: bool = True, **overrides) -> FigureStyle:
    """
    Apply universal Matplotlib/seaborn settings.

    Any FigureStyle field can be overridden for small local tweaks:

        apply_plot_style(font_size=9, panel_height=2.7)

    Set `enforce_size=True` when older notebook cells hard-code `figsize`
    values and you want to force them into the shared sizing system.
    """

    global _CURRENT_STYLE
    style = replace(DEFAULT_STYLE, **overrides) if overrides else DEFAULT_STYLE
    _CURRENT_STYLE = style

    mpl.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "figure.figsize": (style.panel_width, style.panel_height),
            "savefig.dpi": style.dpi,
            "savefig.transparent": style.transparent,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "sans-serif",
            "font.sans-serif": [style.font_family, "DejaVu Sans", "Arial"],
            "font.size": style.font_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "axes.linewidth": style.axis_line_width,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "xtick.major.width": style.tick_width,
            "ytick.major.width": style.tick_width,
            "xtick.major.size": style.tick_length,
            "ytick.major.size": style.tick_length,
            "legend.fontsize": style.legend_size,
            "legend.frameon": False,
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.constrained_layout.use": True,
        }
    )

    try:
        import seaborn as sns

        sns.set_theme(
            context="paper",
            style="ticks",
            font=style.font_family,
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": style.dpi,
                "savefig.dpi": style.dpi,
            },
        )
    except Exception:
        pass

    enforce_universal_figure_size(enforce_size, respect_explicit=respect_explicit_figsize)
    return style


def current_style() -> FigureStyle:
    """Return the currently applied style."""

    return _CURRENT_STYLE


def figure_size(
    nrows: int = 1,
    ncols: int = 1,
    *,
    panel_width: float | None = None,
    panel_height: float | None = None,
    width: float | None = None,
    height: float | None = None,
    style: FigureStyle | Mapping[str, object] | None = None,
) -> tuple[float, float]:
    """Compute a consistent figure size from row/column counts."""

    base = _coerce_style(style)
    if width is None:
        width = ncols * (panel_width or base.panel_width) + max(0, ncols - 1) * base.col_gap
    if height is None:
        height = nrows * (panel_height or base.panel_height) + max(0, nrows - 1) * base.row_gap
    return (float(width), float(height))


def create_figure_grid(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool | str = "row",
    sharey: bool | str = False,
    squeeze: bool = False,
    figsize: tuple[float, float] | None = None,
    style: FigureStyle | Mapping[str, object] | None = None,
    constrained_layout: bool = True,
    **subplot_kwargs,
):
    """
    Create a same-sized, aligned subplot grid.

    `sharex="row"` is the default because panels on the same row, such as E
    and F, should have aligned x axes and equal plot heights.
    """

    base = _coerce_style(style)
    if style is not None:
        apply_plot_style(**asdict(base))
    elif _CURRENT_STYLE == DEFAULT_STYLE:
        apply_plot_style()

    if figsize is None:
        figsize = figure_size(nrows, ncols, style=base)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        constrained_layout=constrained_layout,
        **subplot_kwargs,
    )
    return fig, axes


def enforce_universal_figure_size(enabled: bool = True, *, respect_explicit: bool = False) -> None:
    """
    Optionally force plt.figure/plt.subplots into shared panel dimensions.

    This is useful for older notebooks with hard-coded `figsize` values. By
    default, explicit `figsize` values are overwritten when this patch is
    enabled. Set `respect_explicit=True` to only fill in missing sizes.
    """

    global _SIZE_PATCH_ENABLED

    if not enabled:
        if _SIZE_PATCH_ENABLED:
            plt.figure = _ORIGINAL_PLT_FIGURE
            plt.subplots = _ORIGINAL_PLT_SUBPLOTS
            _SIZE_PATCH_ENABLED = False
        return

    def _patched_figure(*args, **kwargs):
        if "figsize" not in kwargs or not respect_explicit:
            kwargs["figsize"] = figure_size(1, 1)
        return _ORIGINAL_PLT_FIGURE(*args, **kwargs)

    def _patched_subplots(*args, **kwargs):
        nrows = args[0] if len(args) >= 1 else kwargs.get("nrows", 1)
        ncols = args[1] if len(args) >= 2 else kwargs.get("ncols", 1)
        if "figsize" not in kwargs or not respect_explicit:
            kwargs["figsize"] = figure_size(int(nrows), int(ncols))
        return _ORIGINAL_PLT_SUBPLOTS(*args, **kwargs)

    plt.figure = _patched_figure
    plt.subplots = _patched_subplots
    _SIZE_PATCH_ENABLED = True


def make_multipanel_figure(
    layout: Sequence[Sequence[str]],
    *,
    sharex: bool | str = "row",
    sharey: bool | str = False,
    figsize: tuple[float, float] | None = None,
    style: FigureStyle | Mapping[str, object] | None = None,
):
    """
    Create a labeled panel dictionary from a text layout.

    Example:
        fig, panels = make_multipanel_figure([["A", "B"], ["C", "D"]])
        panels["A"].plot(...)
    """

    if not layout or not all(layout):
        raise ValueError("layout must be a non-empty sequence of non-empty rows")
    nrows = len(layout)
    ncols = max(len(row) for row in layout)
    fig, axes = create_figure_grid(
        nrows,
        ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        figsize=figsize,
        style=style,
    )

    panels = {}
    for row_idx, row in enumerate(layout):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(row) or row[col_idx] in {"", ".", None}:
                ax.set_visible(False)
                continue
            label = str(row[col_idx])
            panels[label] = ax
            label_panel(ax, label)
    return fig, panels


def label_panel(
    ax,
    label: str,
    *,
    x: float = -0.14,
    y: float = 1.06,
    fontweight: str = "bold",
    fontsize: float | None = None,
):
    """Place a consistent panel label like A, B, C, etc."""

    style = current_style()
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight=fontweight,
        fontsize=fontsize or style.panel_label_size,
        va="bottom",
        ha="left",
    )
    return ax


def despine(ax=None):
    """Remove top/right spines from one axis or every axis in a figure."""

    if ax is None:
        axes = plt.gcf().axes
    elif isinstance(ax, np.ndarray):
        axes = ax.ravel()
    elif isinstance(ax, (list, tuple)):
        axes = ax
    else:
        axes = [ax]

    for one_ax in axes:
        one_ax.spines["top"].set_visible(False)
        one_ax.spines["right"].set_visible(False)
    return ax


def save_figure(
    fig,
    path: str | Path,
    *,
    formats: Iterable[str] | None = None,
    illustrator: bool = True,
    dpi: int | None = None,
    transparent: bool | None = None,
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
    close: bool = False,
    **savefig_kwargs,
) -> list[Path]:
    """
    Save a figure in consistent export formats.

    Pass a path with or without an extension. For Illustrator editing, keep SVG
    and/or PDF in `formats`; Matplotlib text stays editable through rcParams.
    """

    style = current_style()
    base_path = Path(path)
    if formats is None:
        if base_path.suffix:
            formats = (base_path.suffix.lstrip("."),)
            base_path = base_path.with_suffix("")
        else:
            formats = style.save_formats
    else:
        formats = tuple(fmt.lstrip(".") for fmt in formats)
        if base_path.suffix and base_path.suffix.lstrip(".") not in formats:
            base_path = base_path.with_suffix("")

    if illustrator:
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["svg.fonttype"] = "none"

    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        out_path = base_path.with_suffix(f".{fmt}")
        fig.savefig(
            out_path,
            dpi=dpi or style.dpi,
            transparent=style.transparent if transparent is None else transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **savefig_kwargs,
        )
        saved_paths.append(out_path)

    if close:
        plt.close(fig)
    return saved_paths


def save_current_figure(path: str | Path, **kwargs) -> list[Path]:
    """Save the active Matplotlib figure with `save_figure`."""

    return save_figure(plt.gcf(), path, **kwargs)


@contextmanager
def temporary_plot_style(**overrides):
    """Temporarily apply local tweaks, then restore previous rcParams/style."""

    old_rc = mpl.rcParams.copy()
    old_style = current_style()
    apply_plot_style(**{**asdict(old_style), **overrides})
    try:
        yield current_style()
    finally:
        mpl.rcParams.update(old_rc)
        globals()["_CURRENT_STYLE"] = old_style


def _coerce_style(style: FigureStyle | Mapping[str, object] | None) -> FigureStyle:
    if style is None:
        return current_style()
    if isinstance(style, FigureStyle):
        return style
    if isinstance(style, Mapping):
        return replace(current_style(), **style)
    raise TypeError("style must be a FigureStyle, mapping, or None")


apply_plot_style()
