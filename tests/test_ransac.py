"""Unit tests for RANSAC offset estimation."""

import numpy as np
import pytest

from powermeter_compare.power_comparison import estimate_offset_vs_power_ransac


def test_ransac_linear_relationship_no_outliers():
    """Test RANSAC with clean linear data (no outliers)."""
    # Create synthetic interval results with linear relationship: offset = 0.02 * power + 5
    interval_results = []
    for i, power in enumerate([100, 150, 200, 250, 300], start=1):
        offset = 0.02 * power + 5.0
        interval_results.append(
            {
                "interval_id": i,
                "actual_mean_ref_power": power,
                "offset": offset,
                "std_diff": 2.0,
                "n_samples": 50,
            }
        )

    result = estimate_offset_vs_power_ransac(interval_results, residual_threshold=2.0)

    assert result is not None
    assert result.n_inliers == 5
    assert result.n_outliers == 0
    assert result.slope == pytest.approx(0.02, abs=0.001)
    assert result.intercept == pytest.approx(5.0, abs=0.5)
    assert result.score > 0.99  # Should be near perfect fit


def test_ransac_with_outliers():
    """Test RANSAC robustness with outliers."""
    # Create data with one outlier
    interval_results = [
        {"interval_id": 1, "actual_mean_ref_power": 100, "offset": 7.0, "std_diff": 2.0, "n_samples": 50},
        {"interval_id": 2, "actual_mean_ref_power": 150, "offset": 8.0, "std_diff": 2.0, "n_samples": 50},
        {"interval_id": 3, "actual_mean_ref_power": 200, "offset": 20.0, "std_diff": 2.0, "n_samples": 50},  # Outlier
        {"interval_id": 4, "actual_mean_ref_power": 250, "offset": 10.0, "std_diff": 2.0, "n_samples": 50},
        {"interval_id": 5, "actual_mean_ref_power": 300, "offset": 11.0, "std_diff": 2.0, "n_samples": 50},
    ]

    result = estimate_offset_vs_power_ransac(interval_results, residual_threshold=3.0)

    assert result is not None
    assert result.n_inliers == 4  # Should exclude the outlier
    assert result.n_outliers == 1
    # Outlier should be at index 2
    assert not result.inlier_mask[2]


def test_ransac_insufficient_data():
    """Test RANSAC with insufficient data points."""
    interval_results = [
        {"interval_id": 1, "actual_mean_ref_power": 100, "offset": 7.0, "std_diff": 2.0, "n_samples": 50},
    ]

    result = estimate_offset_vs_power_ransac(interval_results, min_samples=2)

    assert result is None  # Not enough data points


def test_ransac_empty_input():
    """Test RANSAC with empty input."""
    result = estimate_offset_vs_power_ransac([], min_samples=2)

    assert result is None


def test_ransac_constant_offset():
    """Test RANSAC with constant offset (slope near zero)."""
    # All intervals have approximately the same offset regardless of power
    interval_results = []
    np.random.seed(42)
    for i, power in enumerate([100, 150, 200, 250, 300], start=1):
        offset = 8.0 + np.random.normal(0, 0.5)  # Small noise around constant 8W
        interval_results.append(
            {
                "interval_id": i,
                "actual_mean_ref_power": power,
                "offset": offset,
                "std_diff": 1.0,
                "n_samples": 50,
            }
        )

    result = estimate_offset_vs_power_ransac(interval_results, residual_threshold=2.0)

    assert result is not None
    assert result.n_inliers >= 4  # Most points should be inliers
    assert abs(result.slope) < 0.01  # Slope should be near zero
    assert result.intercept == pytest.approx(8.0, abs=1.0)


def test_ransac_power_proportional_offset():
    """Test RANSAC with offset proportional to power (e.g., 3% calibration error)."""
    # offset = 0.03 * power (3% error)
    interval_results = []
    for i, power in enumerate([100, 150, 200, 250, 300], start=1):
        offset = 0.03 * power
        interval_results.append(
            {
                "interval_id": i,
                "actual_mean_ref_power": power,
                "offset": offset,
                "std_diff": 1.0,
                "n_samples": 50,
            }
        )

    result = estimate_offset_vs_power_ransac(interval_results, residual_threshold=2.0)

    assert result is not None
    assert result.n_inliers == 5
    assert result.slope == pytest.approx(0.03, abs=0.001)
    assert abs(result.intercept) < 0.5  # Intercept should be near zero
