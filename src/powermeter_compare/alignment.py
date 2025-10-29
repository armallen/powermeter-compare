"""Module for power data analysis and offset computation."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from powermeter_compare.workout import WorkoutInterval

logger = logging.getLogger(__name__)

# Constants
MIN_SAMPLES_FOR_INTERVAL = 10


def resample_to_common_frequency(
    df: pd.DataFrame,
    time_col: str,
    power_col: str,
    freq: str = "1s",
) -> tuple[pd.Index, pd.Series]:
    """Resample power data to a common frequency.

    DEPRECATED: Use resample_to_common_grid for multi-signal alignment.

    Args:
        df: DataFrame with time and power columns
        time_col: Name of time column
        power_col: Name of power column
        freq: Resampling frequency (default: "1s")

    Returns:
        Tuple of (time_series, power_series)

    """
    df_sorted = df.sort_values(time_col).set_index(time_col)
    power_resampled = df_sorted[power_col].resample(freq).mean()
    return power_resampled.index, power_resampled


def resample_to_common_grid(
    df_ref: pd.DataFrame,
    df_candidate: pd.DataFrame,
    time_col: str = "time",
    power_col: str = "power",
    freq: str = "1s",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample two power signals to a common time grid covering their overlap.

    This function finds the overlapping time period between two recordings and
    resamples both signals to a common time grid, making them suitable for
    direct comparison and correlation analysis.

    Args:
        df_ref: Reference DataFrame with time and power columns
        df_candidate: Candidate DataFrame with time and power columns
        time_col: Name of time column
        power_col: Name of power column
        freq: Resampling frequency (default: "1s")

    Returns:
        Tuple of (df_ref_resampled, df_candidate_resampled) both with same time grid

    """
    # Find the overlapping time range
    ref_start = df_ref[time_col].min()
    ref_end = df_ref[time_col].max()
    cand_start = df_candidate[time_col].min()
    cand_end = df_candidate[time_col].max()

    overlap_start = max(ref_start, cand_start)
    overlap_end = min(ref_end, cand_end)

    if overlap_start >= overlap_end:
        logger.warning("No overlapping time period between reference and candidate signals")
        # Return empty DataFrames
        return pd.DataFrame({time_col: [], power_col: []}), pd.DataFrame({time_col: [], power_col: []})

    overlap_duration = (overlap_end - overlap_start).total_seconds()
    logger.info(f"Overlapping period: {overlap_start} to {overlap_end} ({overlap_duration:.0f}s)")

    # Create common time grid
    common_time_grid = pd.date_range(start=overlap_start, end=overlap_end, freq=freq)

    # Resample reference to common grid
    df_ref_sorted = df_ref.sort_values(time_col).set_index(time_col)
    ref_resampled = df_ref_sorted[power_col].reindex(common_time_grid, method="nearest", tolerance=pd.Timedelta(freq))

    # Resample candidate to common grid
    df_cand_sorted = df_candidate.sort_values(time_col).set_index(time_col)
    cand_resampled = df_cand_sorted[power_col].reindex(common_time_grid, method="nearest", tolerance=pd.Timedelta(freq))

    # Create output DataFrames
    df_ref_out = pd.DataFrame({time_col: common_time_grid, power_col: ref_resampled.to_numpy()})
    df_cand_out = pd.DataFrame({time_col: common_time_grid, power_col: cand_resampled.to_numpy()})

    logger.info(f"Resampled to common grid: {len(common_time_grid)} samples")

    return df_ref_out, df_cand_out


def estimate_time_offset_via_correlation(
    df_ref: pd.DataFrame,
    df_candidate: pd.DataFrame,
    power_col: str = "power",
    max_offset: int = 200,
    *,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict]:
    """Estimate time offset using cross-correlation between power signals.

    IMPORTANT: Input DataFrames should already be resampled to a common time grid
    using resample_to_common_grid(). This function assumes both inputs have the
    same length and aligned timestamps.

    Finds the time lag where the two power signals are most correlated,
    indicating the offset needed to align candidate with reference.

    Args:
        df_ref: Reference power DataFrame (already resampled)
        df_candidate: Candidate power DataFrame (already resampled)
        power_col: Name of power column
        max_offset: Maximum offset to search in seconds
        return_diagnostics: If True, return (offset, diagnostics_dict)

    Returns:
        If return_diagnostics=False: Time offset in seconds to ADD to candidate timestamps
        If return_diagnostics=True: Tuple of (offset, diagnostics_dict) where diagnostics
            contains correlation data for plotting and analysis

    """
    # Extract power values (already aligned from preprocessing)
    ref_vals = df_ref[power_col].fillna(0).to_numpy()
    cand_vals = df_candidate[power_col].fillna(0).to_numpy()

    if len(ref_vals) != len(cand_vals):
        msg = (
            f"Input DataFrames must have same length. Got ref={len(ref_vals)}, "
            f"cand={len(cand_vals)}. Use resample_to_common_grid() first."
        )
        raise ValueError(msg)

    signal_len = len(ref_vals)

    # Constrain max_offset to avoid edge cases with short signals
    # Leave at least 10 samples for correlation
    effective_max_offset = min(max_offset, signal_len - 10)
    if effective_max_offset < 0:
        effective_max_offset = 0
        logger.warning(f"Signal too short ({signal_len}s) for meaningful correlation analysis")

    # Normalize: zero mean, unit variance
    ref_norm = (ref_vals - np.mean(ref_vals)) / (np.std(ref_vals) + 1e-8)
    cand_norm = (cand_vals - np.mean(cand_vals)) / (np.std(cand_vals) + 1e-8)

    # Compute cross-correlation for different lags
    # Positive lag: candidate is delayed (shift candidate backward to align)
    # Negative lag: candidate is ahead (shift candidate forward to align)
    correlations = []
    lags = range(-effective_max_offset, effective_max_offset + 1)

    for lag in lags:
        if lag < 0:
            # Candidate ahead: compare ref[0:len+lag] with cand[-lag:]
            ref_slice = ref_norm[: signal_len + lag]
            cand_slice = cand_norm[-lag:]
        elif lag > 0:
            # Candidate behind: compare ref[lag:] with cand[0:len-lag]
            ref_slice = ref_norm[lag:]
            cand_slice = cand_norm[: signal_len - lag]
        else:
            # No offset
            ref_slice = ref_norm
            cand_slice = cand_norm

        # Compute correlation coefficient
        if len(ref_slice) > 1 and len(cand_slice) > 1:
            corr = np.corrcoef(ref_slice, cand_slice)[0, 1]
            correlations.append(corr if not np.isnan(corr) else -1.0)
        else:
            correlations.append(-1.0)

    # Find lag with maximum correlation
    correlations = np.array(correlations)
    best_idx = np.nanargmax(correlations)
    best_lag = list(lags)[best_idx]
    max_corr = correlations[best_idx]

    # Log top 5 correlation peaks for debugging
    top_indices = np.argsort(correlations)[-5:][::-1]
    logger.debug("Top 5 correlation peaks:")
    for idx in top_indices:
        lag_val = list(lags)[idx]
        corr_val = correlations[idx]
        logger.debug(f"  lag={lag_val:+3d}s, corr={corr_val:.4f}")

    # The offset to apply to candidate timestamps
    offset_to_apply = float(best_lag)

    logger.info(f"Cross-correlation: best_lag={best_lag}s, corr={max_corr:.4f}, offset={offset_to_apply:+.1f}s")

    if return_diagnostics:
        diagnostics = {
            "lags": list(lags),
            "correlations": correlations,
            "best_lag": best_lag,
            "max_corr": max_corr,
            "offset": offset_to_apply,
            "ref_normalized": ref_norm,
            "cand_normalized": cand_norm,
        }
        return offset_to_apply, diagnostics

    return offset_to_apply


def apply_time_offset(df: pd.DataFrame, offset_seconds: float, time_col: str = "time") -> pd.DataFrame:
    """Apply time offset to DataFrame.

    Args:
        df: DataFrame with time column
        offset_seconds: Offset in seconds to apply
        time_col: Name of time column

    Returns:
        DataFrame with adjusted timestamps

    """
    df_shifted = df.copy()
    df_shifted[time_col] = df_shifted[time_col] + timedelta(seconds=offset_seconds)
    return df_shifted


def merge_aligned_data(
    df_ref: pd.DataFrame,
    df_candidate: pd.DataFrame,
    time_col: str = "time",
    tolerance: str = "1s",
) -> pd.DataFrame:
    """Merge time-aligned power data.

    Args:
        df_ref: Reference power DataFrame
        df_candidate: Candidate power DataFrame (already time-aligned)
        time_col: Name of time column
        tolerance: Maximum time difference for matching

    Returns:
        Merged DataFrame with suffixes _ref and _candidate

    """
    merged = pd.merge_asof(
        df_ref.sort_values(time_col),
        df_candidate.sort_values(time_col),
        on=time_col,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
        suffixes=("_ref", "_candidate"),
    )

    n_matched = merged["power_candidate"].notna().sum()
    n_total = len(merged)
    logger.info(f"Merged {n_matched}/{n_total} samples ({100 * n_matched / n_total:.1f}%)")

    return merged


def compute_trailing_averages(df: pd.DataFrame, power_col: str, time_col: str) -> pd.DataFrame:
    """Compute trailing power averages for 5s, 10s, and 30s windows.

    Args:
        df: DataFrame with power and time data
        power_col: Name of power column
        time_col: Name of time column

    Returns:
        DataFrame with additional columns for trailing averages

    """
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
    """Detect regions where power data is missing, zero, or very low.

    Args:
        df: DataFrame with power data
        power_cols: List of power column names to check
        min_power_threshold: Minimum valid power threshold

    Returns:
        Boolean series indicating valid data points

    """
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
        power_1: First power series (reference)
        power_2: Second power series (candidate)
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


def compute_interval_offset(
    df: pd.DataFrame,
    interval: WorkoutInterval,
    start_time: datetime,
    end_time: datetime,
    ref_power_col: str,
    candidate_power_col: str,
    rampup_exclusion_seconds: float = 5.0,
    min_power_threshold: float = 10.0,
) -> dict | None:
    """Compute power offset for a specific workout interval.

    Args:
        df: DataFrame with merged power data and time column
        interval: WorkoutInterval object
        start_time: Actual start time of interval
        end_time: Actual end time of interval
        ref_power_col: Name of reference power column
        candidate_power_col: Name of candidate power column
        rampup_exclusion_seconds: Seconds to exclude from start of interval
        min_power_threshold: Minimum valid power threshold

    Returns:
        Dictionary with offset statistics, or None if insufficient data

    """
    # Apply ramp-up exclusion
    adjusted_start = start_time + pd.Timedelta(seconds=rampup_exclusion_seconds)

    # Filter data to this interval
    mask = (df["time"] >= adjusted_start) & (df["time"] <= end_time)
    interval_data = df[mask]

    if len(interval_data) < MIN_SAMPLES_FOR_INTERVAL:
        logger.warning(f"Insufficient data for interval {interval.interval_id}: {len(interval_data)} samples")
        return None

    # Check for valid power data
    ref_power = interval_data[ref_power_col]
    candidate_power = interval_data[candidate_power_col]

    valid_mask = (
        (ref_power >= min_power_threshold)
        & (candidate_power >= min_power_threshold)
        & ref_power.notna()
        & candidate_power.notna()
    )

    if valid_mask.sum() < MIN_SAMPLES_FOR_INTERVAL:
        logger.warning(f"Insufficient valid data for interval {interval.interval_id}: {valid_mask.sum()} valid samples")
        return None

    # Compute offset
    diff = candidate_power[valid_mask] - ref_power[valid_mask]

    return {
        "interval_id": interval.interval_id,
        "target_power": interval.target_power,
        "target_power_pct": interval.target_power_pct,
        "start_time": start_time,
        "end_time": end_time,
        "duration": interval.duration,
        "rampup_excluded": rampup_exclusion_seconds,
        "offset": np.median(diff),
        "median_diff": np.median(diff),
        "mean_diff": np.mean(diff),
        "std_diff": np.std(diff),
        "actual_mean_ref_power": ref_power[valid_mask].mean(),
        "actual_mean_candidate_power": candidate_power[valid_mask].mean(),
        "n_samples": valid_mask.sum(),
    }


def calculate_mean_std(df: pd.DataFrame, column_name: str) -> tuple[float, float]:
    """Calculate the mean and standard deviation for a column in a DataFrame.

    Args:
        df: DataFrame containing the data
        column_name: Name of the column to analyze

    Returns:
        Tuple of (mean, std)

    """
    mean = np.mean(df[column_name])
    std = np.std(df[column_name])
    return mean, std
