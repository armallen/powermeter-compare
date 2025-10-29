"""Module for creating visualizations of power data."""

import logging
from collections.abc import Sequence
from datetime import datetime

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from powermeter_compare.alignment import calculate_mean_std, compute_offset_only
from powermeter_compare.workout import AlignedIntervals

logger = logging.getLogger(__name__)


def create_time_series_plot(
    df: pd.DataFrame,
    time_col: str,
    series_names: list[str],
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Create time series plot with mean and standard deviation lines.

    Args:
        df: DataFrame with time series data
        time_col: Name of time column
        series_names: List of column names to plot
        labels: Optional mapping of column names to display labels

    Returns:
        Plotly figure object

    """
    labels = labels or {name: name for name in series_names}

    traces = []
    for series in series_names:
        label = labels.get(series, series)
        traces.extend(_create_stat_traces(df, time_col, series, label))

    layout = {"title": "Power Time Series Plot", "xaxis_title": "Time", "yaxis_title": "Power"}
    return go.Figure(data=traces, layout=layout)


def _create_stat_traces(df: pd.DataFrame, time_col: str, series: str, label: str) -> list[go.Scatter]:
    """Create traces for data, mean, and std deviation lines.

    Args:
        df: DataFrame with data
        time_col: Name of time column
        series: Name of data series column
        label: Display label for the series

    Returns:
        List of Plotly scatter traces

    """
    mean, std = calculate_mean_std(df, series)
    time_range = [df[time_col].min(), df[time_col].max()]

    return [
        go.Scatter(x=df[time_col], y=df[series], name=label, mode="lines"),
        go.Scatter(x=time_range, y=[mean, mean], name=f"{label} Mean", line={"color": "red", "dash": "dash"}),
        go.Scatter(x=time_range, y=[mean + std] * 2, name=f"{label} Std", line={"color": "green", "dash": "dot"}),
        go.Scatter(x=time_range, y=[mean - std] * 2, name=f"{label} Std", line={"color": "green", "dash": "dot"}),
    ]


def _find_dropout_regions(valid_mask: pd.Series) -> list[tuple[int, int]]:
    """Find consecutive regions where data is invalid (dropouts).

    Args:
        valid_mask: Boolean series indicating valid data points

    Returns:
        List of (start_idx, end_idx) tuples for dropout regions

    """
    dropout_regions: list[tuple[int, int]] = []
    in_dropout = False
    start_idx = None

    for idx, is_valid in enumerate(valid_mask):
        if not is_valid and not in_dropout:
            in_dropout = True
            start_idx = idx
        elif is_valid and in_dropout:
            assert start_idx is not None
            dropout_regions.append((start_idx, idx - 1))
            in_dropout = False

    if in_dropout:
        assert start_idx is not None
        dropout_regions.append((start_idx, len(valid_mask) - 1))

    return dropout_regions


def _add_dropout_backgrounds(
    fig: go.Figure, df: pd.DataFrame, time_col: str, dropout_regions: list[tuple[int, int]], row: int
) -> None:
    """Add grey background rectangles for dropout regions.

    Args:
        fig: Plotly figure to modify
        df: DataFrame with time data
        time_col: Name of time column
        dropout_regions: List of (start_idx, end_idx) tuples
        row: Subplot row number

    """
    for start_idx, end_idx in dropout_regions:
        if start_idx < len(df) and end_idx < len(df):
            fig.add_vrect(
                x0=df[time_col].iloc[start_idx],
                x1=df[time_col].iloc[end_idx],
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )


def _add_interval_backgrounds(
    fig: go.Figure,
    aligned_intervals: Sequence[tuple[datetime, datetime, object]],
    row: int,
) -> None:
    """Add colored background rectangles for workout intervals.

    Args:
        fig: Plotly figure to modify
        aligned_intervals: List of (start_time, end_time, interval) tuples
        row: Subplot row number

    """
    # Color palette for different intervals
    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "lightpink", "lavender", "peachpuff", "lightcyan"]

    for idx, (start_time, end_time, interval) in enumerate(aligned_intervals):
        if interval.step_type == "E_Normal":
            color = colors[idx % len(colors)]
            fig.add_vrect(
                x0=start_time,
                x1=end_time,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )


def _add_power_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    time_col: str,
    power_cols: tuple[str, str],
    labels: tuple[str, str],
    row: int,
) -> None:
    """Add power traces for both meters to subplot.

    Args:
        fig: Plotly figure to modify
        df: DataFrame with power data
        time_col: Name of time column
        power_cols: Tuple of (ref_power_col, candidate_power_col)
        labels: Tuple of (ref_label, candidate_label)
        row: Subplot row number

    """
    configs = [
        (power_cols[0], labels[0], "blue", "ref"),
        (power_cols[1], labels[1], "red", "candidate"),
    ]
    for power_col, label, color, group in configs:
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[power_col],
                name=label,
                mode="lines",
                line={"color": color, "width": 1.5},
                showlegend=(row == 1),
                legendgroup=group,
            ),
            row=row,
            col=1,
        )


def _add_offset_annotation(
    fig: go.Figure,
    df: pd.DataFrame,
    power_cols: tuple[str, str],
    valid_mask: pd.Series,
    min_power_threshold: float,
    row: int,
) -> None:
    """Compute and add offset annotation to subplot.

    Args:
        fig: Plotly figure to modify
        df: DataFrame with power data
        power_cols: Tuple of (ref_power_col, candidate_power_col)
        valid_mask: Boolean mask for valid data
        min_power_threshold: Minimum power threshold
        row: Subplot row number

    """
    offset_result = compute_offset_only(df[power_cols[0]], df[power_cols[1]], valid_mask, min_power_threshold)
    annotation_text = (
        f"<b>Offset:</b> {offset_result['offset']:.1f}W (std={offset_result['std_diff']:.1f}W, "
        f"n={offset_result['n_samples']})"
    )

    xref = "x domain" if row == 1 else f"x{row} domain"
    yref = "y domain" if row == 1 else f"y{row} domain"

    fig.add_annotation(
        text=annotation_text,
        xref=xref,
        yref=yref,
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        font={"size": 10},
    )


def create_power_comparison_subplots(
    df: pd.DataFrame,
    time_col: str,
    valid_mask: pd.Series,
    label_ref: str,
    label_candidate: str,
    min_power_threshold: float = 50.0,
    aligned_intervals: AlignedIntervals | None = None,
) -> go.Figure:
    """Create subplots comparing power data at different averaging windows.

    Args:
        df: DataFrame with merged power data
        time_col: Name of time column
        valid_mask: Boolean mask for valid data
        label_ref: Label for reference power meter
        label_candidate: Label for candidate power meter
        min_power_threshold: Minimum power threshold
        aligned_intervals: Optional list of workout intervals to highlight

    Returns:
        Plotly figure with subplots

    """
    windows = [("Instantaneous", ""), ("5s Average", "_5s"), ("10s Average", "_10s"), ("30s Average", "_30s")]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[w[0] for w in windows],
        x_title="Time",
        y_title="Power (W)",
    )

    dropout_regions = _find_dropout_regions(valid_mask)

    for row_idx, (_, suffix) in enumerate(windows, start=1):
        power_cols = (f"power_ref{suffix}", f"power_candidate{suffix}")
        labels = (label_ref, label_candidate)

        # Add interval backgrounds first (if provided)
        if aligned_intervals:
            _add_interval_backgrounds(fig, aligned_intervals, row_idx)

        _add_dropout_backgrounds(fig, df, time_col, dropout_regions, row_idx)
        _add_power_traces(fig, df, time_col, power_cols, labels, row_idx)
        _add_offset_annotation(fig, df, power_cols, valid_mask, min_power_threshold, row_idx)
        fig.update_yaxes(title_text="Power (W)", row=row_idx, col=1)

    fig.update_layout(
        height=1200,
        title_text=f"Power Comparison: {label_ref} (ref) vs {label_candidate} (candidate)",
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig


def create_offset_vs_power_plot(
    interval_results: list[dict],
    label_ref: str,
    label_candidate: str,
    ransac_slope: float | None = None,
    ransac_intercept: float | None = None,
) -> go.Figure:
    """Create scatter plot of offset vs target power level.

    Args:
        interval_results: List of interval offset computation results
        label_ref: Label for reference power meter
        label_candidate: Label for candidate power meter
        ransac_slope: Optional RANSAC fit slope to overlay
        ransac_intercept: Optional RANSAC fit intercept to overlay

    Returns:
        Plotly figure

    """
    if not interval_results:
        logger.warning("No interval results to plot")
        return go.Figure()

    # Extract data for plotting
    target_powers = [r["target_power"] for r in interval_results]
    actual_powers = [r["actual_mean_ref_power"] for r in interval_results]
    offsets = [r["offset"] for r in interval_results]
    std_diffs = [r["std_diff"] for r in interval_results]

    fig = go.Figure()

    # Add scatter plot with error bars
    hover_text = [
        f"Target: {tp:.0f}W<br>Actual: {ap:.0f}W<br>Offset: {o:.1f}W"
        for tp, ap, o in zip(target_powers, actual_powers, offsets, strict=True)
    ]
    fig.add_trace(
        go.Scatter(
            x=target_powers,
            y=offsets,
            mode="markers+lines",
            name="Median Offset",
            marker={"size": 10, "color": "blue"},
            error_y={"type": "data", "array": std_diffs, "visible": True},
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Add RANSAC fit line if provided
    if ransac_slope is not None and ransac_intercept is not None:
        power_min = min(actual_powers)
        power_max = max(actual_powers)
        power_range_plot = [power_min - 10, power_max + 10]
        fit_line = [ransac_slope * p + ransac_intercept for p in power_range_plot]

        fig.add_trace(
            go.Scatter(
                x=power_range_plot,
                y=fit_line,
                mode="lines",
                name=f"RANSAC Fit (y={ransac_slope:.4f}x+{ransac_intercept:.2f})",
                line={"color": "green", "width": 2, "dash": "dash"},
            )
        )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero offset")

    fig.update_layout(
        title=f"Power Offset vs Target Power: {label_candidate} - {label_ref}",
        xaxis_title="Target Power (W)",
        yaxis_title="Offset (W)",
        hovermode="closest",
        height=600,
    )

    return fig


def plot_correlation_diagnostics(
    diagnostics: dict,
    label_ref: str = "Reference",
    label_candidate: str = "Candidate",
) -> go.Figure:
    """Create diagnostic plot showing cross-correlation curve.

    Args:
        diagnostics: Diagnostics dictionary from estimate_time_offset_via_correlation
        label_ref: Label for reference power meter
        label_candidate: Label for candidate power meter

    Returns:
        Plotly Figure object

    """
    lags = diagnostics["lags"]
    correlations = diagnostics["correlations"]
    best_lag = diagnostics["best_lag"]
    max_corr = diagnostics["max_corr"]
    offset = diagnostics["offset"]

    fig = go.Figure()

    # Add correlation curve
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=correlations,
            mode="lines",
            name="Correlation",
            line={"color": "blue", "width": 2},
        )
    )

    # Mark the selected offset
    fig.add_trace(
        go.Scatter(
            x=[best_lag],
            y=[max_corr],
            mode="markers",
            name=f"Selected: {offset:+.1f}s",
            marker={"color": "red", "size": 12, "symbol": "star"},
            text=[f"Offset: {offset:+.1f}s<br>Correlation: {max_corr:.4f}"],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Add vertical line at zero lag
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Zero lag")

    fig.update_layout(
        title=f"Cross-Correlation: {label_candidate} vs {label_ref}",
        xaxis_title="Lag (seconds)",
        yaxis_title="Correlation Coefficient",
        hovermode="x unified",
        height=500,
        showlegend=True,
    )

    return fig
