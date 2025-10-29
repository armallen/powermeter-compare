"""Command-line interface for power meter comparison tool.

This module handles argument parsing, validation, logging setup, and
delegates the actual analysis logic to the power_comparison module.
"""

import argparse
import logging
from pathlib import Path

from powermeter_compare.power_comparison import ComparisonConfig, run_comparison

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments.

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If validation fails

    """
    parser = argparse.ArgumentParser(
        description="Compare power data between reference and candidate power meters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required FIT file arguments
    parser.add_argument(
        "--ref-fit",
        type=Path,
        required=True,
        help="Path to the reference FIT file (defines timing baseline)",
    )
    parser.add_argument(
        "--candidate-fit",
        type=Path,
        required=True,
        help="Path to the candidate FIT file (being compared against reference)",
    )

    # Optional labels for power meters
    parser.add_argument(
        "--label-ref",
        type=str,
        default=None,
        help="Label for reference power meter (auto-detected if not provided)",
    )
    parser.add_argument(
        "--label-candidate",
        type=str,
        default=None,
        help="Label for candidate power meter (auto-detected if not provided)",
    )

    # Workout-based analysis arguments
    parser.add_argument(
        "--workout-json",
        type=Path,
        default=None,
        help="Path to MyWhoosh workout JSON file for interval-based analysis",
    )
    parser.add_argument(
        "--ftp",
        type=float,
        default=None,
        help="Functional Threshold Power (watts) - required if --workout-json is provided",
    )
    parser.add_argument(
        "--rampup-exclusion",
        type=float,
        default=5.0,
        help="Seconds to exclude from start of each interval (default: 5s)",
    )

    # Analysis parameters
    parser.add_argument(
        "--min-power-threshold",
        type=float,
        default=50.0,
        help="Minimum power threshold for analysis (watts)",
    )
    parser.add_argument(
        "--time-tolerance",
        type=str,
        default="2s",
        help="Maximum time difference for GPS timestamp alignment (e.g., '2s', '1s')",
    )
    parser.add_argument(
        "--max-time-offset",
        type=int,
        default=60,
        help="Maximum time offset to search in cross-correlation (seconds)",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate parsed arguments.

    Args:
        args: Parsed arguments

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If argument combinations are invalid

    """
    # Validate file existence
    if not args.ref_fit.exists():
        msg = f"Reference FIT file does not exist: {args.ref_fit}"
        raise FileNotFoundError(msg)

    if not args.candidate_fit.exists():
        msg = f"Candidate FIT file does not exist: {args.candidate_fit}"
        raise FileNotFoundError(msg)

    # Validate workout-related arguments
    if args.workout_json:
        if not args.workout_json.exists():
            msg = f"Workout JSON file does not exist: {args.workout_json}"
            raise FileNotFoundError(msg)
        if args.ftp is None:
            msg = "FTP must be provided when using --workout-json"
            raise ValueError(msg)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress verbose matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def main() -> None:
    """Run the CLI application."""
    # Setup logging
    setup_logging()

    # Parse and validate arguments
    args = parse_arguments()

    try:
        validate_arguments(args)
    except (FileNotFoundError, ValueError) as e:
        logger.exception("Validation error")
        raise SystemExit(1) from e

    # Create configuration
    config = ComparisonConfig(
        ref_fit=args.ref_fit,
        candidate_fit=args.candidate_fit,
        label_ref=args.label_ref,
        label_candidate=args.label_candidate,
        workout_json=args.workout_json,
        ftp=args.ftp,
        rampup_exclusion=args.rampup_exclusion,
        min_power_threshold=args.min_power_threshold,
        time_tolerance=args.time_tolerance,
        max_time_offset=args.max_time_offset,
    )

    # Run analysis
    try:
        run_comparison(config)
        logger.info("Analysis completed successfully")
    except Exception:
        logger.exception("Analysis failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
