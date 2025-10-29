"""Unit tests for alignment module functions."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from powermeter_compare.alignment import (
    resample_to_common_grid,
    apply_time_offset,
    merge_aligned_data,
    compute_trailing_averages,
    detect_dropout_regions,
    compute_offset_only,
    compute_interval_offset,
    calculate_mean_std,
)
from powermeter_compare.workout import WorkoutInterval


@pytest.fixture
def sample_power_df():
    """Create a sample power DataFrame."""
    times = pd.date_range("2024-01-01 10:00:00", periods=60, freq="1s")
    power = np.concatenate([np.full(20, 100), np.full(20, 200), np.full(20, 150)])
    return pd.DataFrame({"time": times, "power": power})


@pytest.fixture
def sample_interval():
    """Create a sample workout interval."""
    return WorkoutInterval(
        interval_id=1,
        step_type="E_Normal",
        start_time=10.0,
        end_time=30.0,
        duration=20.0,
        target_power=200.0,
        target_power_pct=0.8,
    )


def test_resample_to_common_grid_overlapping():
    """Test resampling with overlapping time periods."""
    # Reference: 0-60s
    times_ref = pd.date_range("2024-01-01 10:00:00", periods=60, freq="1s")
    power_ref = np.full(60, 200)
    df_ref = pd.DataFrame({"time": times_ref, "power": power_ref})

    # Candidate: 10-70s (10s offset)
    times_cand = pd.date_range("2024-01-01 10:00:10", periods=60, freq="1s")
    power_cand = np.full(60, 210)
    df_cand = pd.DataFrame({"time": times_cand, "power": power_cand})

    df_ref_res, df_cand_res = resample_to_common_grid(df_ref, df_cand)

    # Should have 50 samples (overlap from 10s to 59s)
    assert len(df_ref_res) == len(df_cand_res)
    assert len(df_ref_res) == 50
    assert df_ref_res["time"].min() == times_cand[0]
    assert df_ref_res["time"].max() == times_ref[-1]


def test_resample_to_common_grid_no_overlap():
    """Test resampling with no overlap returns empty DataFrames."""
    times_ref = pd.date_range("2024-01-01 10:00:00", periods=30, freq="1s")
    df_ref = pd.DataFrame({"time": times_ref, "power": np.full(30, 200)})

    times_cand = pd.date_range("2024-01-01 10:05:00", periods=30, freq="1s")
    df_cand = pd.DataFrame({"time": times_cand, "power": np.full(30, 200)})

    df_ref_res, df_cand_res = resample_to_common_grid(df_ref, df_cand)

    assert len(df_ref_res) == 0
    assert len(df_cand_res) == 0


def test_apply_time_offset_positive():
    """Test applying positive time offset."""
    times = pd.date_range("2024-01-01 10:00:00", periods=10, freq="1s")
    df = pd.DataFrame({"time": times, "power": np.full(10, 200)})

    df_shifted = apply_time_offset(df, 5.0)

    assert (df_shifted["time"] - df["time"]).dt.total_seconds().iloc[0] == 5.0
    assert len(df_shifted) == len(df)


def test_apply_time_offset_negative():
    """Test applying negative time offset."""
    times = pd.date_range("2024-01-01 10:00:00", periods=10, freq="1s")
    df = pd.DataFrame({"time": times, "power": np.full(10, 200)})

    df_shifted = apply_time_offset(df, -3.5)

    assert (df_shifted["time"] - df["time"]).dt.total_seconds().iloc[0] == -3.5


def test_merge_aligned_data():
    """Test merging aligned power data."""
    times_ref = pd.date_range("2024-01-01 10:00:00", periods=30, freq="1s")
    df_ref = pd.DataFrame({"time": times_ref, "power": np.full(30, 200)})

    times_cand = pd.date_range("2024-01-01 10:00:00", periods=30, freq="1s")
    df_cand = pd.DataFrame({"time": times_cand, "power": np.full(30, 210)})

    merged = merge_aligned_data(df_ref, df_cand)

    assert len(merged) == 30
    assert "power_ref" in merged.columns
    assert "power_candidate" in merged.columns
    assert merged["power_ref"].iloc[0] == 200
    assert merged["power_candidate"].iloc[0] == 210


def test_merge_aligned_data_with_gap():
    """Test merging with time gaps respects tolerance."""
    times_ref = pd.date_range("2024-01-01 10:00:00", periods=10, freq="1s")
    df_ref = pd.DataFrame({"time": times_ref, "power": np.arange(10)})

    # Candidate has 5s gap
    times_cand = pd.date_range("2024-01-01 10:00:05", periods=10, freq="1s")
    df_cand = pd.DataFrame({"time": times_cand, "power": np.arange(10, 20)})

    merged = merge_aligned_data(df_ref, df_cand, tolerance="2s")

    # With 2s tolerance, last 5 samples should not match
    assert merged["power_candidate"].notna().sum() < len(merged)


def test_compute_trailing_averages(sample_power_df):
    """Test computation of trailing averages."""
    result = compute_trailing_averages(sample_power_df, "power", "time")

    assert "power_5s" in result.columns
    assert "power_10s" in result.columns
    assert "power_30s" in result.columns
    assert len(result) == len(sample_power_df)

    # Check that trailing averages are computed
    assert result["power_30s"].notna().any()


def test_detect_dropout_regions_no_dropouts():
    """Test dropout detection with valid data."""
    df = pd.DataFrame({"power_ref": np.full(30, 200), "power_candidate": np.full(30, 210)})

    valid_mask = detect_dropout_regions(df, ["power_ref", "power_candidate"], min_power_threshold=10.0)

    assert valid_mask.all()


def test_detect_dropout_regions_with_zeros():
    """Test dropout detection with zero power values."""
    df = pd.DataFrame(
        {
            "power_ref": np.concatenate([np.full(10, 200), np.zeros(10), np.full(10, 200)]),
            "power_candidate": np.full(30, 210),
        }
    )

    valid_mask = detect_dropout_regions(df, ["power_ref", "power_candidate"], min_power_threshold=10.0)

    assert valid_mask.sum() == 20  # Only non-zero regions


def test_detect_dropout_regions_with_nans():
    """Test dropout detection with NaN values."""
    power_ref = np.full(30, 200.0)
    power_ref[10:15] = np.nan

    df = pd.DataFrame({"power_ref": power_ref, "power_candidate": np.full(30, 210)})

    valid_mask = detect_dropout_regions(df, ["power_ref", "power_candidate"], min_power_threshold=10.0)

    assert valid_mask.sum() == 25  # Exclude NaN region


def test_compute_offset_only_basic():
    """Test basic offset computation."""
    power_ref = pd.Series(np.full(30, 200))
    power_cand = pd.Series(np.full(30, 210))
    valid_mask = pd.Series(np.full(30, True))

    result = compute_offset_only(power_ref, power_cand, valid_mask, min_power_threshold=50.0)

    assert result["offset"] == pytest.approx(10.0)
    assert result["median_diff"] == pytest.approx(10.0)
    assert result["n_samples"] == 30


def test_compute_offset_only_with_invalid_data():
    """Test offset computation excluding invalid data."""
    power_ref = pd.Series(np.concatenate([np.full(10, 200), np.zeros(10), np.full(10, 200)]))
    power_cand = pd.Series(np.concatenate([np.full(10, 210), np.zeros(10), np.full(10, 210)]))
    valid_mask = pd.Series(np.concatenate([np.full(10, True), np.full(10, False), np.full(10, True)]))

    result = compute_offset_only(power_ref, power_cand, valid_mask, min_power_threshold=50.0)

    assert result["offset"] == pytest.approx(10.0)
    assert result["n_samples"] == 20  # Only valid data


def test_compute_interval_offset_valid(sample_interval):
    """Test interval offset computation with valid data."""
    times = pd.date_range("2024-01-01 10:00:00", periods=60, freq="1s")
    power_ref = np.concatenate([np.full(20, 100), np.full(20, 200), np.full(20, 150)])
    power_cand = np.concatenate([np.full(20, 105), np.full(20, 208), np.full(20, 155)])

    df = pd.DataFrame({"time": times, "power_ref": power_ref, "power_candidate": power_cand})

    start_time = times[10]
    end_time = times[29]

    result = compute_interval_offset(
        df, sample_interval, start_time, end_time, "power_ref", "power_candidate", rampup_exclusion_seconds=0.0
    )

    assert result is not None
    assert result["interval_id"] == 1
    # Offset is median of [5,5,...,5,8,8,...,8] = 6.5
    assert result["offset"] == pytest.approx(6.5, abs=0.1)
    assert result["n_samples"] == 20


def test_compute_interval_offset_insufficient_data(sample_interval):
    """Test interval offset with insufficient data returns None."""
    times = pd.date_range("2024-01-01 10:00:00", periods=20, freq="1s")
    df = pd.DataFrame({"time": times, "power_ref": np.full(20, 200), "power_candidate": np.full(20, 210)})

    start_time = times[0]
    end_time = times[4]  # Only 5 samples

    result = compute_interval_offset(
        df, sample_interval, start_time, end_time, "power_ref", "power_candidate", rampup_exclusion_seconds=0.0
    )

    assert result is None


def test_compute_interval_offset_with_rampup_exclusion(sample_interval):
    """Test interval offset with ramp-up exclusion."""
    times = pd.date_range("2024-01-01 10:00:00", periods=60, freq="1s")
    power_ref = np.full(60, 200)
    power_cand = np.full(60, 210)

    df = pd.DataFrame({"time": times, "power_ref": power_ref, "power_candidate": power_cand})

    start_time = times[10]
    end_time = times[39]

    result = compute_interval_offset(
        df, sample_interval, start_time, end_time, "power_ref", "power_candidate", rampup_exclusion_seconds=5.0
    )

    assert result is not None
    assert result["rampup_excluded"] == 5.0
    # Should have 25 samples (30 total - 5 rampup)
    assert result["n_samples"] == 25


def test_calculate_mean_std():
    """Test mean and standard deviation calculation."""
    df = pd.DataFrame({"power": [100, 200, 150, 180, 170]})

    mean, std = calculate_mean_std(df, "power")

    assert mean == pytest.approx(160.0)
    assert std == pytest.approx(np.std([100, 200, 150, 180, 170]))


def test_calculate_mean_std_constant_values():
    """Test mean and std with constant values."""
    df = pd.DataFrame({"power": np.full(50, 200)})

    mean, std = calculate_mean_std(df, "power")

    assert mean == 200.0
    assert std == 0.0
