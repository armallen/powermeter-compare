import argparse
from pathlib import Path

import fitparse
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np


# Define a function to read a FIT file and return a pandas DataFrame
def read_fit_file(file_path: Path):
    fitfile = fitparse.FitFile(str(file_path))
    # Iterate over all messages in the FIT file
    df = pd.DataFrame(
        [record.get_values() for record in fitfile.get_messages("record")]
    )
    assert "power" in df.columns, "No power data in FIT file."
    assert "timestamp" in df.columns, "No timestamp data in FIT file."

    df = df.sort_values(by="timestamp")
    return df


def power_from_df(df: pd.DataFrame, power_coefficient: float = 1.0) -> pd.DataFrame:
    # Create a pandas DataFrame from the power and time data
    pdf = pd.DataFrame({"power": df["power"], "time": df["timestamp"]})

    # Convert the time column to a datetime format
    pdf["time"] = pd.to_datetime(df["timestamp"], unit="s")
    pdf["power"] = power_smoothing(pdf["power"]) * power_coefficient
    return pdf


def power_smoothing(series: pd.Series, window_size: int = 3) -> pd.Series:
    return series.rolling(window_size).mean()


def plot_plotly(df: pd.DataFrame):
    # plot a scatter plot with relevant statistics using Plotly Express
    fig = px.scatter(
        df,
        x="power_1",
        y="power_2",
        trendline="ols",
        labels={"power_1": "Power 1", "power_2": "Power 2"},
        title="Scatter Plot of Power 1 vs Power 2",
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

    # Parse the arguments
    args = parser.parse_args()
    assert args.fit_1.exists() and args.fit_2.exists(), "Both input files must exist."
    return args


def create_time_series_plot(
    df: pd.DataFrame, time_col: str, series_names: list[str]
) -> go.Figure:
    traces = []
    for series in series_names:
        mean, std = calculate_mean_std(df, series)
        traces.extend(
            [
                go.Scatter(x=df[time_col], y=df[series], name=series, mode="lines"),
                go.Scatter(
                    x=[min(df[time_col]), max(df[time_col])],
                    y=[mean, mean],
                    name=f"{series} Mean",
                    line=dict(color="red", dash="dash"),
                ),
                go.Scatter(
                    x=[min(df[time_col]), max(df[time_col])],
                    y=[mean + std, mean + std],
                    name=f"{series} Std",
                    line=dict(color="green", dash="dot"),
                ),
                go.Scatter(
                    x=[min(df[time_col]), max(df[time_col])],
                    y=[mean - std, mean + -std],
                    name=f"{series} Std",
                    line=dict(color="green", dash="dot"),
                ),
            ]
        )

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title="Power Time Series Plot",
            xaxis_title="Time",
            yaxis_title="Power",
        ),
    )


if __name__ == "__main__":
    args = handle_arguments()

    # Read the two FIT files into pandas DataFrames
    df_1 = power_from_df(read_fit_file(args.fit_1))
    df_2 = power_from_df(read_fit_file(args.fit_2))

    # Merge the two DataFrames on the time column
    merged_df = pd.merge(df_1, df_2, on="time", suffixes=("_1", "_2"))

    # Create a plot of the power values over time
    plt.plot(merged_df["time"], merged_df["power_1"], label="File 1")
    plt.plot(merged_df["time"], merged_df["power_2"], label="File 2")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()

    plot_plotly(merged_df)
    fig = create_time_series_plot(merged_df, "time", ["power_1", "power_2"])
    fig.show()
