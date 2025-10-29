"""Unit tests for plotting module helper functions."""

import numpy as np
import pandas as pd
import pytest

from powermeter_compare.plotting import (
    _create_stat_traces,
    _find_dropout_regions,
    create_offset_vs_power_plot,
    create_power_comparison_subplots,
    create_time_series_plot,
    plot_correlation_diagnostics,
)
from powermeter_compare.workout import WorkoutInterval


def test_find_dropout_regions_no_dropouts():
    """Test dropout detection with no dropouts."""
    valid_mask = pd.Series([True] * 30)

    regions = _find_dropout_regions(valid_mask)

    assert len(regions) == 0


def test_find_dropout_regions_single_dropout():
    """Test dropout detection with single dropout region."""
    valid_mask = pd.Series([True] * 10 + [False] * 5 + [True] * 10)

    regions = _find_dropout_regions(valid_mask)

    assert len(regions) == 1
    assert regions[0] == (10, 14)


def test_find_dropout_regions_multiple_dropouts():
    """Test dropout detection with multiple dropout regions."""
    valid_mask = pd.Series([True] * 5 + [False] * 3 + [True] * 5 + [False] * 4 + [True] * 5)

    regions = _find_dropout_regions(valid_mask)

    assert len(regions) == 2
    assert regions[0] == (5, 7)
    assert regions[1] == (13, 16)


def test_find_dropout_regions_at_start():
    """Test dropout at the start of data."""
    valid_mask = pd.Series([False] * 5 + [True] * 10)

    regions = _find_dropout_regions(valid_mask)

    assert len(regions) == 1
    assert regions[0] == (0, 4)


def test_find_dropout_regions_at_end():
    """Test dropout at the end of data."""
    valid_mask = pd.Series([True] * 10 + [False] * 5)

    regions = _find_dropout_regions(valid_mask)

    assert len(regions) == 1
    assert regions[0] == (10, 14)


def test_find_dropout_regions_entire_invalid():
    """Test with entirely invalid data."""
    valid_mask = pd.Series([False] * 20)

    regions = _find_dropout_regions(valid_mask)

    assert len(regions) == 1
    assert regions[0] == (0, 19)


def test_create_stat_traces():
    """Test creation of statistical traces."""
    times = pd.date_range("2024-01-01 10:00:00", periods=30, freq="1s")
    power = np.full(30, 200)
    df = pd.DataFrame({"time": times, "power": power})

    traces = _create_stat_traces(df, "time", "power", "Test Power")

    # Should create 4 traces: data, mean, +std, -std
    assert len(traces) == 4
    assert traces[0].name == "Test Power"
    assert traces[1].name == "Test Power Mean"


def test_create_time_series_plot():
    """Test time series plot creation."""
    times = pd.date_range("2024-01-01 10:00:00", periods=30, freq="1s")
    df = pd.DataFrame({"time": times, "power_ref": np.full(30, 200), "power_cand": np.full(30, 210)})

    fig = create_time_series_plot(df, "time", ["power_ref", "power_cand"])

    assert fig is not None
    assert len(fig.data) > 0  # Should have traces


def test_create_time_series_plot_with_labels():
    """Test time series plot with custom labels."""
    times = pd.date_range("2024-01-01 10:00:00", periods=30, freq="1s")
    df = pd.DataFrame({"time": times, "power_ref": np.full(30, 200), "power_cand": np.full(30, 210)})

    labels = {"power_ref": "Reference PM", "power_cand": "Candidate PM"}
    fig = create_time_series_plot(df, "time", ["power_ref", "power_cand"], labels=labels)

    assert fig is not None


def test_create_power_comparison_subplots():
    """Test power comparison subplot creation."""
    times = pd.date_range("2024-01-01 10:00:00", periods=60, freq="1s")
    power_ref = np.full(60, 200)
    power_cand = np.full(60, 208)

    df = pd.DataFrame(
        {
            "time": times,
            "power_ref": power_ref,
            "power_candidate": power_cand,
            "power_ref_5s": power_ref,
            "power_candidate_5s": power_cand,
            "power_ref_10s": power_ref,
            "power_candidate_10s": power_cand,
            "power_ref_30s": power_ref,
            "power_candidate_30s": power_cand,
        }
    )

    valid_mask = pd.Series([True] * 60)

    fig = create_power_comparison_subplots(df, "time", valid_mask, "Ref PM", "Cand PM")

    assert fig is not None
    # Should have 4 subplots (instantaneous, 5s, 10s, 30s)
    assert len(fig.data) > 0


def test_create_offset_vs_power_plot():
    """Test offset vs power plot creation."""
    interval_results = [
        {"target_power": 150, "actual_mean_ref_power": 148, "offset": 5.0, "std_diff": 2.0},
        {"target_power": 200, "actual_mean_ref_power": 198, "offset": 6.5, "std_diff": 2.5},
        {"target_power": 250, "actual_mean_ref_power": 248, "offset": 7.0, "std_diff": 3.0},
    ]

    fig = create_offset_vs_power_plot(interval_results, "Ref PM", "Cand PM")

    assert fig is not None
    assert len(fig.data) > 0


def test_create_offset_vs_power_plot_empty():
    """Test offset plot with empty results."""
    fig = create_offset_vs_power_plot([], "Ref PM", "Cand PM")

    assert fig is not None
    # Empty figure should still be created


def test_plot_correlation_diagnostics():
    """Test correlation diagnostics plot creation."""
    diagnostics = {
        "lags": list(range(-10, 11)),
        "correlations": np.random.rand(21),
        "best_lag": 3,
        "max_corr": 0.95,
        "offset": -3.0,
    }

    fig = plot_correlation_diagnostics(diagnostics, "Ref PM", "Cand PM")

    assert fig is not None
    assert len(fig.data) > 0


def test_plot_correlation_diagnostics_with_negative_offset():
    """Test correlation diagnostics with negative offset."""
    diagnostics = {
        "lags": list(range(-20, 21)),
        "correlations": np.concatenate([np.random.rand(15), [0.98], np.random.rand(25)]),
        "best_lag": -5,
        "max_corr": 0.98,
        "offset": 5.0,
    }

    fig = plot_correlation_diagnostics(diagnostics, "Ref PM", "Cand PM")

    assert fig is not None
    # Should have traces for correlation curve and selected point
    assert len(fig.data) >= 2


def test_create_power_comparison_subplots_with_intervals():
    """Test power comparison with workout intervals."""
    times = pd.date_range("2024-01-01 10:00:00", periods=60, freq="1s")
    power_ref = np.full(60, 200)
    power_cand = np.full(60, 208)

    df = pd.DataFrame(
        {
            "time": times,
            "power_ref": power_ref,
            "power_candidate": power_cand,
            "power_ref_5s": power_ref,
            "power_candidate_5s": power_cand,
            "power_ref_10s": power_ref,
            "power_candidate_10s": power_cand,
            "power_ref_30s": power_ref,
            "power_candidate_30s": power_cand,
        }
    )

    valid_mask = pd.Series([True] * 60)

    # Create mock interval

    interval = WorkoutInterval(1, "E_Normal", 10.0, 30.0, 20.0, 200.0, 0.8)
    aligned_intervals = [(times[10], times[29], interval)]

    fig = create_power_comparison_subplots(
        df,
        "time",
        valid_mask,
        "Ref PM",
        "Cand PM",
        aligned_intervals=aligned_intervals,
    )

    assert fig is not None
