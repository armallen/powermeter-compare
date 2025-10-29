"""Module for reading and processing FIT files."""

import logging
from pathlib import Path

import fitparse
import pandas as pd

logger = logging.getLogger(__name__)


def fit_file_to_records_df(file_path: Path) -> pd.DataFrame:
    """Read FIT file and return DataFrame with record messages.

    Args:
        file_path: Path to the FIT file

    Returns:
        DataFrame with power and timestamp columns, sorted by timestamp

    """
    fitfile = fitparse.FitFile(str(file_path))
    df = pd.DataFrame([record.get_values() for record in fitfile.get_messages("record")])

    assert "power" in df.columns, "No power data in FIT file."
    assert "timestamp" in df.columns, "No timestamp data in FIT file."

    return df.sort_values(by="timestamp")


def fit_messages_to_df(file_path: Path) -> pd.DataFrame:
    """Convert all FIT file messages to a pandas DataFrame.

    Args:
        file_path: Path to the FIT file

    Returns:
        DataFrame with all messages and their types

    """
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
    """Extract power meter device information from FIT file.

    Args:
        file_path: Path to the FIT file

    Returns:
        Formatted power meter name string

    """
    fitfile = fitparse.FitFile(str(file_path))
    for device_info in fitfile.get_messages("device_info"):
        values = device_info.get_values()
        if values.get("antplus_device_type") == "bike_power":
            logger.debug(f"Found power meter device: {values}")
            return _format_power_meter_name(values)

    return "Power Meter"


def power_from_df(df: pd.DataFrame, power_coefficient: float = 1.0) -> pd.DataFrame:
    """Extract power and time data from FIT records.

    Args:
        df: DataFrame with FIT record data
        power_coefficient: Multiplier for power values

    Returns:
        DataFrame with smoothed power and datetime columns

    """
    return pd.DataFrame(
        {
            "power": power_smoothing(df["power"]) * power_coefficient,
            "time": pd.to_datetime(df["timestamp"], unit="s"),
        }
    )


def power_smoothing(series: pd.Series, window_size: int = 3) -> pd.Series:
    """Apply rolling average smoothing to power data.

    Args:
        series: Power data series
        window_size: Rolling window size in samples

    Returns:
        Smoothed power series

    """
    return series.rolling(window_size).mean()
