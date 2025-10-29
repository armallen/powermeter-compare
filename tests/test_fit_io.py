"""Unit tests for FIT file I/O module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from powermeter_compare.fit_io import (
    _format_power_meter_name,
    power_from_df,
    power_smoothing,
)


def test_format_power_meter_name_with_product_name():
    """Test formatting with product name."""
    values = {"manufacturer": "Garmin", "product_name": "Vector 3", "product": 12345}

    result = _format_power_meter_name(values)

    assert result == "Garmin Vector 3"


def test_format_power_meter_name_with_product_code():
    """Test formatting with product code only."""
    values = {"manufacturer": "Wahoo", "product": 54321}

    result = _format_power_meter_name(values)

    assert result == "Wahoo (54321)"


def test_format_power_meter_name_minimal():
    """Test formatting with only manufacturer."""
    values = {"manufacturer": "Stages"}

    result = _format_power_meter_name(values)

    assert result == "Stages Power Meter"


def test_format_power_meter_name_unknown():
    """Test formatting with no manufacturer."""
    values: dict = {}

    result = _format_power_meter_name(values)

    assert result == "Unknown Power Meter"


def test_power_smoothing_basic():
    """Test basic power smoothing."""
    power = pd.Series([100, 200, 300, 200, 100])

    smoothed = power_smoothing(power, window_size=3)

    # First two values will be NaN due to window
    assert pd.isna(smoothed.iloc[0])
    assert pd.isna(smoothed.iloc[1])
    # Third value should be average of first 3
    assert smoothed.iloc[2] == pytest.approx(200.0)


def test_power_smoothing_window_size_1():
    """Test smoothing with window size 1 (no smoothing)."""
    power = pd.Series([100, 200, 300, 200, 100])

    smoothed = power_smoothing(power, window_size=1)

    # First value will be NaN with rolling window
    # But actually, with window=1, should have values
    assert len(smoothed) == len(power)


def test_power_smoothing_constant_values():
    """Test smoothing with constant power values."""
    power = pd.Series(np.full(10, 200))

    smoothed = power_smoothing(power, window_size=3)

    # All non-NaN values should be 200
    assert smoothed.dropna().eq(200).all()


def test_power_from_df_basic():
    """Test power extraction from FIT records DataFrame."""
    timestamps = np.arange(1704110400, 1704110430, 1)  # 30 seconds of timestamps
    power = np.full(30, 200)

    df = pd.DataFrame({"timestamp": timestamps, "power": power})

    result = power_from_df(df)

    assert "power" in result.columns
    assert "time" in result.columns
    assert len(result) == 30
    # Check datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(result["time"])


def test_power_from_df_with_coefficient():
    """Test power extraction with scaling coefficient."""
    timestamps = np.arange(1704110400, 1704110410, 1)
    power = np.full(10, 200)

    df = pd.DataFrame({"timestamp": timestamps, "power": power})

    result = power_from_df(df, power_coefficient=1.05)

    # Power should be scaled (after smoothing which introduces NaNs)
    # Check the non-NaN values
    assert result["power"].dropna().iloc[-1] == pytest.approx(210.0)


def test_power_from_df_with_variable_power():
    """Test with realistic variable power data."""
    timestamps = np.arange(1704110400, 1704110420, 1)  # 20 seconds
    power = np.array(
        [100, 150, 200, 250, 200, 180, 160, 150, 140, 130, 120, 150, 180, 200, 220, 240, 230, 210, 190, 170]
    )

    df = pd.DataFrame({"timestamp": timestamps, "power": power})

    result = power_from_df(df)

    assert len(result) == 20
    # Smoothing should reduce variability
    assert result["power"].notna().any()
