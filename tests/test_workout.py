"""Unit tests for workout parsing module."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from powermeter_compare.workout import (
    WorkoutInterval,
    Workout,
    parse_mywhoosh_workout,
    align_intervals_to_timestamps,
)


@pytest.fixture
def sample_workout_json(tmp_path):
    """Create a temporary workout JSON file."""
    workout_data = {
        "Name": "Test Workout",
        "Description": "A test workout with intervals",
        "Time": 300,
        "WorkoutStepsArray": [
            {"Id": 1, "StepType": "E_Normal", "Time": 60, "Power": 0.6},  # 60s @ 60% FTP
            {"Id": 2, "StepType": "E_Normal", "Time": 120, "Power": 0.8},  # 120s @ 80% FTP
            {"Id": 3, "StepType": "E_Normal", "Time": 60, "Power": 1.0},  # 60s @ 100% FTP
            {"Id": 4, "StepType": "E_FreeRide", "Time": 60, "Power": 0.0},  # 60s free ride
        ],
    }

    json_file = tmp_path / "workout.json"
    json_file.write_text(json.dumps(workout_data))
    return json_file


def test_workout_interval_dataclass():
    """Test WorkoutInterval dataclass creation."""
    interval = WorkoutInterval(
        interval_id=1,
        step_type="E_Normal",
        start_time=0.0,
        end_time=60.0,
        duration=60.0,
        target_power=200.0,
        target_power_pct=0.8,
    )

    assert interval.interval_id == 1
    assert interval.duration == 60.0
    assert interval.target_power == 200.0


def test_workout_dataclass():
    """Test Workout dataclass creation."""
    intervals = [
        WorkoutInterval(1, "E_Normal", 0.0, 60.0, 60.0, 150.0, 0.6),
        WorkoutInterval(2, "E_Normal", 60.0, 180.0, 120.0, 200.0, 0.8),
    ]

    workout = Workout(
        name="Test",
        description="Test workout",
        total_time=180,
        intervals=intervals,
    )

    assert workout.name == "Test"
    assert len(workout.intervals) == 2
    assert workout.total_time == 180


def test_workout_constant_power_intervals():
    """Test filtering constant power intervals."""
    intervals = [
        WorkoutInterval(1, "E_Normal", 0.0, 60.0, 60.0, 150.0, 0.6),
        WorkoutInterval(2, "E_FreeRide", 60.0, 120.0, 60.0, None, None),
        WorkoutInterval(3, "E_Normal", 120.0, 180.0, 60.0, 200.0, 0.8),
    ]

    workout = Workout("Test", "Test", 180, intervals)

    constant_intervals = workout.constant_power_intervals()

    assert len(constant_intervals) == 2
    assert all(i.step_type == "E_Normal" for i in constant_intervals)
    assert all(i.target_power is not None for i in constant_intervals)


def test_workout_get_intervals_at_time():
    """Test getting intervals at specific time."""
    intervals = [
        WorkoutInterval(1, "E_Normal", 0.0, 60.0, 60.0, 150.0, 0.6),
        WorkoutInterval(2, "E_Normal", 60.0, 180.0, 120.0, 200.0, 0.8),
        WorkoutInterval(3, "E_Normal", 180.0, 240.0, 60.0, 250.0, 1.0),
    ]

    workout = Workout("Test", "Test", 240, intervals)

    # At 30s, should be in interval 1
    intervals_at_30 = workout.get_intervals_at_time(30.0)
    assert len(intervals_at_30) == 1
    assert intervals_at_30[0].interval_id == 1

    # At 60s, should be at boundary (end of interval 1, start of interval 2)
    intervals_at_60 = workout.get_intervals_at_time(60.0)
    assert len(intervals_at_60) == 2

    # At 120s, should be in interval 2
    intervals_at_120 = workout.get_intervals_at_time(120.0)
    assert len(intervals_at_120) == 1
    assert intervals_at_120[0].interval_id == 2


def test_parse_mywhoosh_workout(sample_workout_json):
    """Test parsing MyWhoosh workout JSON."""
    ftp = 250.0

    workout = parse_mywhoosh_workout(sample_workout_json, ftp)

    assert workout.name == "Test Workout"
    assert workout.description == "A test workout with intervals"
    assert len(workout.intervals) == 4
    assert workout.total_time == 300

    # Check first interval
    interval_1 = workout.intervals[0]
    assert interval_1.interval_id == 1
    assert interval_1.start_time == 0.0
    assert interval_1.end_time == 60.0
    assert interval_1.duration == 60.0
    assert interval_1.target_power == pytest.approx(150.0)  # 60% of 250W
    assert interval_1.target_power_pct == pytest.approx(0.6)

    # Check second interval
    interval_2 = workout.intervals[1]
    assert interval_2.start_time == 60.0
    assert interval_2.end_time == 180.0
    assert interval_2.target_power == pytest.approx(200.0)  # 80% of 250W


def test_parse_mywhoosh_workout_with_freeride(sample_workout_json):
    """Test parsing workout with FreeRide intervals."""
    ftp = 250.0

    workout = parse_mywhoosh_workout(sample_workout_json, ftp)

    # Last interval is FreeRide
    freeride_interval = workout.intervals[3]
    assert freeride_interval.step_type == "E_FreeRide"
    assert freeride_interval.target_power is None
    assert freeride_interval.target_power_pct is None


def test_align_intervals_to_timestamps():
    """Test aligning workout intervals to absolute timestamps."""
    intervals = [
        WorkoutInterval(1, "E_Normal", 0.0, 60.0, 60.0, 150.0, 0.6),
        WorkoutInterval(2, "E_Normal", 60.0, 180.0, 120.0, 200.0, 0.8),
        WorkoutInterval(3, "E_Normal", 180.0, 240.0, 60.0, 250.0, 1.0),
    ]

    reference_start = datetime(2024, 1, 1, 10, 0, 0)

    aligned = align_intervals_to_timestamps(intervals, reference_start)

    assert len(aligned) == 3

    # Check first interval
    start_1, end_1, interval_1 = aligned[0]
    assert start_1 == reference_start
    assert end_1 == reference_start + timedelta(seconds=60)
    assert interval_1.interval_id == 1

    # Check second interval
    start_2, end_2, interval_2 = aligned[1]
    assert start_2 == reference_start + timedelta(seconds=60)
    assert end_2 == reference_start + timedelta(seconds=180)
    assert interval_2.interval_id == 2


def test_align_intervals_to_timestamps_different_start_time():
    """Test alignment with different reference start time."""
    intervals = [
        WorkoutInterval(1, "E_Normal", 0.0, 30.0, 30.0, 150.0, 0.6),
    ]

    reference_start = datetime(2024, 6, 15, 14, 30, 0)

    aligned = align_intervals_to_timestamps(intervals, reference_start)

    start, end, _ = aligned[0]
    assert start == reference_start
    assert end == reference_start + timedelta(seconds=30)


def test_parse_mywhoosh_workout_empty_steps(tmp_path):
    """Test parsing workout with no steps."""
    workout_data = {
        "Name": "Empty Workout",
        "Description": "No steps",
        "Time": 0,
        "WorkoutStepsArray": [],
    }

    json_file = tmp_path / "empty_workout.json"
    json_file.write_text(json.dumps(workout_data))

    workout = parse_mywhoosh_workout(json_file, ftp=250.0)

    assert workout.name == "Empty Workout"
    assert len(workout.intervals) == 0
    assert workout.total_time == 0
