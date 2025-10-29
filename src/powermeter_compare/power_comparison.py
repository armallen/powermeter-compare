"""Power meter comparison analysis pipeline.

This module contains the core logic for comparing power data between two power meters.
It orchestrates the entire analysis workflow from loading FIT files to generating plots.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor

from powermeter_compare.alignment import (
    apply_time_offset,
    compute_interval_offset,
    compute_trailing_averages,
    detect_dropout_regions,
    estimate_time_offset_via_correlation,
    merge_aligned_data,
    resample_to_common_grid,
)
from powermeter_compare.fit_io import fit_file_to_records_df, get_power_meter_info, power_from_df
from powermeter_compare.plotting import (
    create_offset_vs_power_plot,
    create_power_comparison_subplots,
    plot_correlation_diagnostics,
)
from powermeter_compare.workout import Workout, WorkoutInterval, align_intervals_to_timestamps, parse_mywhoosh_workout

logger = logging.getLogger(__name__)


@dataclass
class PowerData:
    """Container for power meter data."""

    df: pd.DataFrame
    label: str
    start_time: datetime


@dataclass
class AlignmentResult:
    """Container for time alignment results."""

    time_offset: float
    df_aligned: pd.DataFrame
    correlation_diagnostics: dict | None


@dataclass
class MergedData:
    """Container for merged and processed power data."""

    df: pd.DataFrame
    valid_mask: pd.Series
    power_cols: list[str]


@dataclass
class ComparisonConfig:
    """Configuration for power comparison analysis."""

    ref_fit: Path
    candidate_fit: Path
    label_ref: str | None = None
    label_candidate: str | None = None
    workout_json: Path | None = None
    ftp: float | None = None
    rampup_exclusion: float = 5.0
    min_power_threshold: float = 50.0
    time_tolerance: str = "2s"
    max_time_offset: int = 60


def load_fit_files(ref_fit_path: Path, candidate_fit_path: Path) -> tuple[PowerData, PowerData]:
    """Load and process FIT files for both power meters.

    Args:
        ref_fit_path: Path to reference FIT file
        candidate_fit_path: Path to candidate FIT file

    Returns:
        Tuple of (reference_data, candidate_data)

    """
    logger.info("Step 1: Reading FIT files...")

    # Load reference
    label_ref = get_power_meter_info(ref_fit_path)
    df_ref = power_from_df(fit_file_to_records_df(ref_fit_path))
    ref_data = PowerData(df=df_ref, label=label_ref, start_time=df_ref["time"].min())

    # Load candidate
    label_candidate = get_power_meter_info(candidate_fit_path)
    df_candidate = power_from_df(fit_file_to_records_df(candidate_fit_path))
    cand_data = PowerData(df=df_candidate, label=label_candidate, start_time=df_candidate["time"].min())

    logger.info(f"  Reference ({label_ref}) start: {ref_data.start_time}")
    logger.info(f"  Candidate ({label_candidate}) start: {cand_data.start_time}")

    return ref_data, cand_data


def align_power_data(ref_data: PowerData, cand_data: PowerData, max_offset: int = 60) -> AlignmentResult:
    """Align candidate power data to reference using cross-correlation.

    Args:
        ref_data: Reference power data
        cand_data: Candidate power data
        max_offset: Maximum time offset to search (seconds)

    Returns:
        AlignmentResult with offset and aligned candidate data

    """
    # Step 2: Resample to common grid
    logger.info("\nStep 2: Resampling to common time grid...")
    df_ref_resampled, df_candidate_resampled = resample_to_common_grid(
        ref_data.df, cand_data.df, time_col="time", power_col="power", freq="1s"
    )

    # Step 3: Estimate time offset
    logger.info("\nStep 3: Estimating time offset via cross-correlation...")
    time_offset, correlation_diagnostics = estimate_time_offset_via_correlation(
        df_ref_resampled,
        df_candidate_resampled,
        power_col="power",
        max_offset=max_offset,
        return_diagnostics=True,
    )

    # Step 4: Apply offset
    logger.info(f"\nStep 4: Applying time offset ({time_offset:+.1f}s) to candidate data...")
    df_candidate_aligned = apply_time_offset(cand_data.df, time_offset, time_col="time")

    aligned_start = df_candidate_aligned["time"].min()
    time_delta = (aligned_start - ref_data.start_time).total_seconds()
    logger.info(f"  Candidate start after alignment: {aligned_start}")
    logger.info(f"  Time delta after alignment: {time_delta:.2f}s")

    return AlignmentResult(
        time_offset=time_offset, df_aligned=df_candidate_aligned, correlation_diagnostics=correlation_diagnostics
    )


def prepare_merged_data(
    ref_df: pd.DataFrame, candidate_aligned_df: pd.DataFrame, time_tolerance: str = "2s"
) -> MergedData:
    """Merge aligned data and prepare for analysis.

    Args:
        ref_df: Reference power DataFrame
        candidate_aligned_df: Time-aligned candidate DataFrame
        time_tolerance: Maximum time difference for merging

    Returns:
        MergedData with merged DataFrame and valid data mask

    """
    # Step 5: Merge
    logger.info("\nStep 5: Merging aligned data...")
    merged_df = merge_aligned_data(ref_df, candidate_aligned_df, time_col="time", tolerance=time_tolerance)

    # Step 6: Detect dropouts
    logger.info("\nStep 6: Detecting dropout regions...")
    power_cols = ["power_ref", "power_candidate"]
    valid_mask = detect_dropout_regions(merged_df, power_cols, min_power_threshold=10.0)

    valid_pct = 100 * valid_mask.sum() / len(valid_mask)
    logger.info(f"  Valid data: {valid_mask.sum()} / {len(valid_mask)} ({valid_pct:.1f}%)")

    # Set invalid to NaN
    merged_df.loc[~valid_mask, "power_ref"] = np.nan
    merged_df.loc[~valid_mask, "power_candidate"] = np.nan

    # Step 7: Compute trailing averages
    logger.info("\nStep 7: Computing trailing averages...")
    merged_df = compute_trailing_averages(merged_df, "power_ref", "time")
    merged_df = compute_trailing_averages(merged_df, "power_candidate", "time")

    return MergedData(df=merged_df, valid_mask=valid_mask, power_cols=power_cols)


def analyze_intervals(
    merged_data: MergedData,
    workout: Workout,
    reference_start_time: datetime,
    rampup_exclusion_seconds: float = 5.0,
) -> tuple[list[tuple[datetime, datetime, WorkoutInterval]], list[dict]]:
    """Analyze workout intervals and compute per-interval offsets.

    Args:
        merged_data: Merged power data
        workout: Parsed workout definition
        reference_start_time: Start time from reference FIT file
        rampup_exclusion_seconds: Seconds to exclude from interval start

    Returns:
        Tuple of (aligned_intervals, interval_results)

    """
    logger.info("\nStep 8: Analyzing workout intervals...")
    logger.info(f"  Workout: {workout.name}")
    logger.info(f"  Total time: {workout.total_time}s")

    # Get constant power intervals
    constant_power_intervals = workout.constant_power_intervals()
    logger.info(f"  Found {len(constant_power_intervals)} constant power intervals")

    # Align to timestamps
    aligned_intervals = align_intervals_to_timestamps(constant_power_intervals, reference_start_time)

    # Compute per-interval offsets
    logger.info("\n  Computing per-interval offsets:")
    interval_results = []

    for start_time, end_time, interval in aligned_intervals:
        result = compute_interval_offset(
            df=merged_data.df,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            ref_power_col="power_ref",
            candidate_power_col="power_candidate",
            rampup_exclusion_seconds=rampup_exclusion_seconds,
            min_power_threshold=10.0,
        )
        if result:
            interval_results.append(result)
            logger.info(
                f"  Interval {result['interval_id']}: "
                f"Target={result['target_power']:.0f}W, "
                f"Offset={result['offset']:.1f}W ± {result['std_diff']:.1f}W, "
                f"n={result['n_samples']}"
            )

    # Display summary
    if interval_results:
        _log_interval_summary(interval_results)

    return aligned_intervals, interval_results


def _log_interval_summary(interval_results: list[dict]) -> None:
    """Log formatted summary table of interval results.

    Args:
        interval_results: List of interval offset computation results

    """
    logger.info("\n=== Interval Offset Summary ===")
    logger.info(f"{'Target (W)':<12} {'Actual (W)':<12} {'Offset (W)':<12} {'Std (W)':<10} {'Samples':<8}")
    logger.info("-" * 64)

    for result in interval_results:
        logger.info(
            f"{result['target_power']:<12.0f} "
            f"{result['actual_mean_ref_power']:<12.1f} "
            f"{result['offset']:<12.1f} "
            f"{result['std_diff']:<10.1f} "
            f"{result['n_samples']:<8}"
        )


@dataclass
class RansacFitResult:
    """Container for RANSAC regression results."""

    slope: float
    intercept: float
    inlier_mask: np.ndarray
    n_inliers: int
    n_outliers: int
    score: float
    power_range: tuple[float, float]


def estimate_offset_vs_power_ransac(
    interval_results: list[dict],
    residual_threshold: float = 2.0,
    min_samples: int = 2,
) -> RansacFitResult | None:
    """Estimate offset as a function of power using RANSAC regression.

    Uses RANSAC (RANdom SAmple Consensus) to robustly fit a linear model
    relating power offset to power level, handling outliers automatically.

    Args:
        interval_results: List of interval offset computation results
        residual_threshold: Maximum residual for a sample to be classified as inlier (watts)
        min_samples: Minimum number of samples required to fit model

    Returns:
        RansacFitResult with model parameters and diagnostics, or None if insufficient data

    """
    if not interval_results or len(interval_results) < min_samples:
        logger.warning(f"Insufficient interval data for RANSAC: {len(interval_results)} intervals")
        return None

    # Extract power and offset data
    powers = np.array([r["actual_mean_ref_power"] for r in interval_results])
    offsets = np.array([r["offset"] for r in interval_results])

    # Reshape for sklearn (needs 2D array)
    x = powers.reshape(-1, 1)
    y = offsets

    # Fit RANSAC regressor
    ransac = RANSACRegressor(
        residual_threshold=residual_threshold,
        min_samples=min_samples,
        random_state=42,
    )

    try:
        ransac.fit(x, y)
    except ValueError:
        logger.exception("RANSAC fitting failed")
        return None

    # Extract results
    inlier_mask = ransac.inlier_mask_
    n_inliers = int(np.sum(inlier_mask))
    n_outliers = int(np.sum(~inlier_mask))

    # Get model parameters (linear: offset = slope * power + intercept)
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    # Compute R² score on inliers
    score = ransac.score(x[inlier_mask], y[inlier_mask])

    power_range = (float(powers.min()), float(powers.max()))

    logger.info("\n=== RANSAC Offset vs Power Model ===")
    logger.info(f"  Model: offset = {slope:.4f} * power + {intercept:.2f}")
    logger.info(f"  Inliers: {n_inliers}/{len(interval_results)} ({100 * n_inliers / len(interval_results):.0f}%)")
    logger.info(f"  Outliers: {n_outliers}")
    logger.info(f"  R² score: {score:.4f}")
    logger.info(f"  Power range: {power_range[0]:.0f}W - {power_range[1]:.0f}W")

    # Log outliers if any
    if n_outliers > 0:
        logger.info("\n  Outlier intervals:")
        for i, is_outlier in enumerate(~inlier_mask):
            if is_outlier:
                result = interval_results[i]
                predicted_offset = slope * result["actual_mean_ref_power"] + intercept
                residual = result["offset"] - predicted_offset
                logger.info(
                    f"    Interval {result['interval_id']}: "
                    f"Power={result['actual_mean_ref_power']:.0f}W, "
                    f"Offset={result['offset']:.1f}W, "
                    f"Predicted={predicted_offset:.1f}W, "
                    f"Residual={residual:.1f}W"
                )

    return RansacFitResult(
        slope=slope,
        intercept=intercept,
        inlier_mask=inlier_mask,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        score=score,
        power_range=power_range,
    )


def generate_plots(
    merged_data: MergedData,
    label_ref: str,
    label_candidate: str,
    correlation_diagnostics: dict | None,
    min_power_threshold: float = 50.0,
    aligned_intervals: list[tuple[datetime, datetime, WorkoutInterval]] | None = None,
    interval_results: list[dict] | None = None,
    ransac_fit: RansacFitResult | None = None,
) -> None:
    """Generate and display all visualization plots.

    Args:
        merged_data: Merged power data
        label_ref: Reference power meter label
        label_candidate: Candidate power meter label
        correlation_diagnostics: Correlation diagnostics from alignment
        min_power_threshold: Minimum power threshold for analysis
        aligned_intervals: Optional workout intervals
        interval_results: Optional per-interval offset results
        ransac_fit: Optional RANSAC fit results for offset vs power

    """
    # Main comparison plot
    logger.info("Creating comprehensive power comparison plots")
    fig_main = create_power_comparison_subplots(
        merged_data.df,
        time_col="time",
        valid_mask=merged_data.valid_mask,
        label_ref=label_ref,
        label_candidate=label_candidate,
        min_power_threshold=min_power_threshold,
        aligned_intervals=aligned_intervals,
    )
    fig_main.show()

    # Offset vs power plot (with RANSAC fit if available)
    if interval_results:
        logger.info("Creating offset vs power level plot")
        fig_offset = create_offset_vs_power_plot(
            interval_results,
            label_ref,
            label_candidate,
            ransac_slope=ransac_fit.slope if ransac_fit else None,
            ransac_intercept=ransac_fit.intercept if ransac_fit else None,
        )
        fig_offset.show()

    # Correlation diagnostics plot
    if correlation_diagnostics:
        logger.info("Creating cross-correlation diagnostic plot")
        fig_corr = plot_correlation_diagnostics(correlation_diagnostics, label_ref, label_candidate)
        fig_corr.show()


def run_comparison(config: ComparisonConfig) -> None:
    """Run complete power meter comparison analysis.

    This is the main orchestrator function that coordinates all analysis steps.

    Args:
        config: Configuration parameters for the comparison

    """
    # Load FIT files
    ref_data, cand_data = load_fit_files(config.ref_fit, config.candidate_fit)

    # Override labels if provided
    if config.label_ref:
        ref_data.label = config.label_ref
    if config.label_candidate:
        cand_data.label = config.label_candidate

    logger.info(f"Processing {ref_data.label} (reference) vs {cand_data.label} (candidate)")

    # Align power data
    alignment = align_power_data(ref_data, cand_data, max_offset=config.max_time_offset)

    # Prepare merged data
    merged_data = prepare_merged_data(ref_data.df, alignment.df_aligned, time_tolerance=config.time_tolerance)

    # Analyze intervals if workout provided
    aligned_intervals = None
    interval_results = []
    ransac_fit = None

    if config.workout_json and config.ftp:
        workout = parse_mywhoosh_workout(config.workout_json, config.ftp)
        aligned_intervals, interval_results = analyze_intervals(
            merged_data, workout, ref_data.start_time, rampup_exclusion_seconds=config.rampup_exclusion
        )

        # Perform RANSAC analysis if we have interval results
        if interval_results:
            ransac_fit = estimate_offset_vs_power_ransac(interval_results)

    # Generate plots
    generate_plots(
        merged_data,
        ref_data.label,
        cand_data.label,
        alignment.correlation_diagnostics,
        min_power_threshold=config.min_power_threshold,
        aligned_intervals=aligned_intervals,
        interval_results=interval_results,
        ransac_fit=ransac_fit,
    )
