import argparse
import logging
from pathlib import Path

import fitparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def fit_file_to_records_df(file_path: Path) -> pd.DataFrame:
    """Read FIT file and return DataFrame with record messages."""
    fitfile = fitparse.FitFile(str(file_path))
    df = pd.DataFrame([record.get_values() for record in fitfile.get_messages("record")])

    assert "power" in df.columns, "No power data in FIT file."
    assert "timestamp" in df.columns, "No timestamp data in FIT file."

    return df.sort_values(by="timestamp")


def fit_messages_to_df(file_path: Path) -> pd.DataFrame:
    """Convert all FIT file messages to a pandas DataFrame."""
    fitfile = fitparse.FitFile(str(file_path))
    messages = []
    for message in fitfile.get_messages():
        msg_data = message.get_values()
        msg_data["message_type"] = message.name
        messages.append(msg_data)

    return pd.DataFrame(messages)


def _format_power_meter_name(values: dict) -> str:
    """Format power meter device name from device info values."""
    manufacturer = values.get("manufacturer", "Unknown")
    product_name = values.get("product_name", "")
    product = values.get("product", "")

    if product_name:
        return f"{manufacturer} {product_name}"
    if product:
        return f"{manufacturer} ({product})"
    return f"{manufacturer} Power Meter"


def get_power_meter_info(file_path: Path) -> str:
    """Extract power meter device information from FIT file."""
    fitfile = fitparse.FitFile(str(file_path))
    for device_info in fitfile.get_messages("device_info"):
        values = device_info.get_values()
        if values.get("antplus_device_type") == "bike_power":
            logger.debug(f"Found power meter device: {values}")
            return _format_power_meter_name(values)

    return "Power Meter"


def power_from_df(df: pd.DataFrame, power_coefficient: float = 1.0) -> pd.DataFrame:
    """Extract power and time data from FIT records."""
    return pd.DataFrame(
        {
            "power": power_smoothing(df["power"]) * power_coefficient,
            "time": pd.to_datetime(df["timestamp"], unit="s"),
        }
    )


def power_smoothing(series: pd.Series, window_size: int = 3) -> pd.Series:
    return series.rolling(window_size).mean()


def compute_trailing_averages(df: pd.DataFrame, power_col: str, time_col: str) -> pd.DataFrame:
    """Compute trailing power averages for 5s, 10s, and 30s windows."""
    result_df = df.copy().sort_values(by=time_col)

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(result_df[time_col]):
        result_df[time_col] = pd.to_datetime(result_df[time_col])

    # Set time as index for rolling operations
    result_df = result_df.set_index(time_col)

    # Compute trailing averages using time-based windows
    for window in ["5s", "10s", "30s"]:
        result_df[f"{power_col}_{window}"] = result_df[power_col].rolling(window, closed="left").mean()

    return result_df.reset_index()


def detect_dropout_regions(df: pd.DataFrame, power_cols: list[str], min_power_threshold: float = 10.0) -> pd.Series:
    """Detect regions where power data is missing, zero, or very low."""
    valid_mask = pd.Series(data=True, index=df.index)
    for col in power_cols:
        valid_mask &= df[col].notna() & (df[col] >= min_power_threshold)
    return valid_mask


def compute_offset_only(
    power_1: pd.Series,
    power_2: pd.Series,
    valid_mask: pd.Series,
    min_power_threshold: float = 50.0,
) -> dict:
    """Compute offset-only estimate using median difference.

    Args:
        power_1: First power series
        power_2: Second power series
        valid_mask: Boolean mask for valid data points
        min_power_threshold: Only use data above this threshold

    Returns:
        Dictionary with offset estimate and statistics

    """
    # Filter valid data above threshold
    mask = valid_mask & (power_1 >= min_power_threshold) & (power_2 >= min_power_threshold)

    # Compute difference: power_2 - power_1
    diff = power_2[mask] - power_1[mask]

    return {
        "offset": np.median(diff),
        "median_diff": np.median(diff),
        "mean_diff": np.mean(diff),
        "std_diff": np.std(diff),
        "n_samples": mask.sum(),
    }


def plot_plotly(df: pd.DataFrame, label_1: str = "Power 1", label_2: str = "Power 2") -> None:
    """Create scatter plot with correlation coefficient."""
    fig = px.scatter(
        df,
        x="power_1",
        y="power_2",
        trendline="ols",
        labels={"power_1": label_1, "power_2": label_2},
        title=f"Scatter Plot of {label_1} vs {label_2}",
    )
    corr_coef = df["power_1"].corr(df["power_2"])
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.1,
        text=f"Correlation coefficient: {corr_coef:.2f}",
        showarrow=False,
    )
    fig.show()


def calculate_mean_std(df: pd.DataFrame, column_name: str) -> tuple[float, float]:
    """Calculate the mean and standard deviation for a column in a Pandas DataFrame."""
    mean = np.mean(df[column_name])
    std = np.std(df[column_name])
    return mean, std


def handle_arguments() -> argparse.Namespace:
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process two file paths.")

    # Add the required file path arguments
    parser.add_argument("--fit-1", type=Path, help="Path to the first file.")
    parser.add_argument("--fit-2", type=Path, help="Path to the second file.")

    # Add optional labels for power meters
    parser.add_argument("--label-1", type=str, default=None, help="Label for the first power meter.")
    parser.add_argument("--label-2", type=str, default=None, help="Label for the second power meter.")

    # Parse the arguments
    args = parser.parse_args()
    assert args.fit_1.exists(), f"First input file does not exist: {args.fit_1}"
    assert args.fit_2.exists(), f"Second input file does not exist: {args.fit_2}"
    return args


def _create_stat_traces(df: pd.DataFrame, time_col: str, series: str, label: str) -> list[go.Scatter]:
    """Create traces for data, mean, and std deviation lines."""
    mean, std = calculate_mean_std(df, series)
    time_range = [df[time_col].min(), df[time_col].max()]

    return [
        go.Scatter(x=df[time_col], y=df[series], name=label, mode="lines"),
        go.Scatter(x=time_range, y=[mean, mean], name=f"{label} Mean", line={"color": "red", "dash": "dash"}),
        go.Scatter(x=time_range, y=[mean + std] * 2, name=f"{label} Std", line={"color": "green", "dash": "dot"}),
        go.Scatter(x=time_range, y=[mean - std] * 2, name=f"{label} Std", line={"color": "green", "dash": "dot"}),
    ]


def create_time_series_plot(
    df: pd.DataFrame,
    time_col: str,
    series_names: list[str],
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Create time series plot with mean and standard deviation lines."""
    labels = labels or {name: name for name in series_names}

    traces = []
    for series in series_names:
        label = labels.get(series, series)
        traces.extend(_create_stat_traces(df, time_col, series, label))

    layout = {"title": "Power Time Series Plot", "xaxis_title": "Time", "yaxis_title": "Power"}
    return go.Figure(data=traces, layout=layout)


def _find_dropout_regions(valid_mask: pd.Series) -> list[tuple[int, int]]:
    """Find consecutive regions where data is invalid (dropouts)."""
    dropout_regions = []
    in_dropout = False
    start_idx = None

    for idx, is_valid in enumerate(valid_mask):
        if not is_valid and not in_dropout:
            in_dropout = True
            start_idx = idx
        elif is_valid and in_dropout:
            dropout_regions.append((start_idx, idx - 1))
            in_dropout = False

    if in_dropout:
        dropout_regions.append((start_idx, len(valid_mask) - 1))

    return dropout_regions


def _add_dropout_backgrounds(
    fig: go.Figure, df: pd.DataFrame, time_col: str, dropout_regions: list[tuple[int, int]], row: int
) -> None:
    """Add grey background rectangles for dropout regions."""
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


def _add_power_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    time_col: str,
    power_cols: tuple[str, str],
    labels: tuple[str, str],
    row: int,
) -> None:
    """Add power traces for both meters to subplot."""
    configs = [
        (power_cols[0], labels[0], "blue", "power1"),
        (power_cols[1], labels[1], "red", "power2"),
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
    """Compute and add offset annotation to subplot."""
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
    label_1: str,
    label_2: str,
    min_power_threshold: float = 50.0,
) -> go.Figure:
    """Create subplots comparing power data at different averaging windows."""
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
        power_cols = (f"power_1{suffix}", f"power_2{suffix}")
        labels = (label_1, label_2)

        _add_dropout_backgrounds(fig, df, time_col, dropout_regions, row_idx)
        _add_power_traces(fig, df, time_col, power_cols, labels, row_idx)
        _add_offset_annotation(fig, df, power_cols, valid_mask, min_power_threshold, row_idx)
        fig.update_yaxes(title_text="Power (W)", row=row_idx, col=1)

    fig.update_layout(
        height=1200,
        title_text=f"Power Comparison: {label_1} vs {label_2}",
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Silence matplotlib debug logs
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    args = handle_arguments()

    # Determine labels for the power meters
    label_1 = args.label_1 or get_power_meter_info(args.fit_1)
    label_2 = args.label_2 or get_power_meter_info(args.fit_2)

    logger.info(f"Processing {label_1} vs {label_2}")

    # Read the two FIT files into pandas DataFrames
    df_1 = power_from_df(fit_file_to_records_df(args.fit_1))
    df_2 = power_from_df(fit_file_to_records_df(args.fit_2))

    # Merge the two DataFrames on the time column FIRST
    merged_df = df_1.merge(df_2, on="time", suffixes=("_1", "_2"))
    logger.info(f"Merged data has {len(merged_df)} records")

    # Detect dropout regions (where either meter has low/zero power)
    power_cols = ["power_1", "power_2"]
    valid_mask = detect_dropout_regions(merged_df, power_cols, min_power_threshold=10.0)
    logger.info(
        f"Valid data points: {valid_mask.sum()} / {len(valid_mask)} ({100 * valid_mask.sum() / len(valid_mask):.1f}%)"
    )

    # Set invalid values to NaN for both meters so they're excluded from rolling averages
    merged_df.loc[~valid_mask, "power_1"] = np.nan
    merged_df.loc[~valid_mask, "power_2"] = np.nan

    # Now compute trailing averages on the cleaned data
    # pandas rolling will automatically exclude NaN values from the calculation
    logger.info("Computing trailing averages for both power meters")
    merged_df = compute_trailing_averages(merged_df, "power_1", "time")
    merged_df = compute_trailing_averages(merged_df, "power_2", "time")

    # Create comprehensive subplot figure with all analyses
    logger.info("Creating comprehensive power comparison plots")
    fig = create_power_comparison_subplots(
        merged_df,
        time_col="time",
        valid_mask=valid_mask,
        label_1=label_1,
        label_2=label_2,
        min_power_threshold=50.0,
    )
    fig.show()

    # Also create the scatter plot for correlation analysis
    logger.info("Creating scatter plot for correlation analysis")
    plot_plotly(merged_df, label_1, label_2)
