"""Module for parsing and processing workout files.

Handles MyWhoosh workout JSON files and converts them to usable interval data.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class WorkoutInterval:
    """Represents a single workout interval with power target."""

    interval_id: int
    step_type: str  # E_Normal, E_FreeRide, etc.
    start_time: float  # Seconds from workout start
    end_time: float  # Seconds from workout start
    duration: float  # Seconds
    target_power: float | None  # Watts (or None for FreeRide)
    target_power_pct: float | None  # FTP percentage (or None)


@dataclass
class Workout:
    """Represents a complete workout with metadata and intervals."""

    name: str
    description: str
    total_time: int  # Total workout duration in seconds
    intervals: list[WorkoutInterval]

    def constant_power_intervals(self) -> list[WorkoutInterval]:
        """Return only intervals with constant power targets (E_Normal)."""
        return [i for i in self.intervals if i.step_type == "E_Normal" and i.target_power is not None]

    def get_intervals_at_time(self, elapsed_seconds: float) -> list[WorkoutInterval]:
        """Return all intervals that are active at the given time."""
        return [i for i in self.intervals if i.start_time <= elapsed_seconds <= i.end_time]


def parse_mywhoosh_workout(file_path: Path, ftp: float) -> Workout:
    """Parse a MyWhoosh workout JSON file.

    Args:
        file_path: Path to the workout JSON file
        ftp: Functional Threshold Power in watts

    Returns:
        Workout object with all intervals

    """
    data = json.loads(file_path.read_text())

    intervals = []
    current_time = 0.0

    for step in data.get("WorkoutStepsArray", []):
        step_type = step.get("StepType", "E_Normal")
        duration = step.get("Time", 0)
        power_pct = step.get("Power", 0.0)

        # Convert power percentage to watts (0.47 -> 47% FTP)
        target_power = power_pct * ftp if power_pct > 0 else None
        target_power_pct = power_pct if power_pct > 0 else None

        interval = WorkoutInterval(
            interval_id=step.get("Id"),
            step_type=step_type,
            start_time=current_time,
            end_time=current_time + duration,
            duration=duration,
            target_power=target_power,
            target_power_pct=target_power_pct,
        )
        intervals.append(interval)
        current_time += duration

    return Workout(
        name=data.get("Name", "Unknown"),
        description=data.get("Description", ""),
        total_time=data.get("Time", int(current_time)),
        intervals=intervals,
    )


AlignedInterval = tuple[datetime, datetime, WorkoutInterval]
AlignedIntervals = list[AlignedInterval]


def align_intervals_to_timestamps(intervals: list[WorkoutInterval], reference_start_time: datetime) -> AlignedIntervals:
    """Convert workout intervals to absolute timestamp ranges.

    Args:
        intervals: List of workout intervals with relative times
        reference_start_time: The start timestamp from the reference FIT file

    Returns:
        List of tuples: (start_datetime, end_datetime, interval)

    """
    aligned = []
    for interval in intervals:
        start_dt = reference_start_time + timedelta(seconds=interval.start_time)
        end_dt = reference_start_time + timedelta(seconds=interval.end_time)
        aligned.append((start_dt, end_dt, interval))
    return aligned
