# powermeter-compare

A Python tool for comparing power data between two cycling power meters using FIT files. Performs automatic time alignment, detects dropout regions, computes power offsets, and generates interactive visualizations.

## Features

- **Automatic Time Alignment**: Uses cross-correlation to align candidate power meter data with reference data
- **Dropout Detection**: Identifies and excludes regions with missing, zero, or invalid power readings
- **Power Offset Analysis**: Computes median power offset between meters with statistical metrics
- **Interval-Based Analysis**: Optional per-interval offset computation using MyWhoosh workout files
- **RANSAC Regression**: Robust linear modeling of offset as a function of power (detects systematic calibration errors)
- **Interactive Visualizations**: Plotly-based plots with synchronized zoom and detailed diagnostics

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Clone the repository
git clone https://github.com/yourusername/powermeter-compare.git
cd powermeter-compare

# Install dependencies
uv sync

# Run the tool
uv run powermeter-compare --help
```

## Quick Start

### Basic Comparison

Compare two power meters using their FIT files:

```bash
uv run powermeter-compare \
  --ref-fit data/reference.fit \
  --candidate-fit data/candidate.fit
```

### Interval-Based Analysis

Analyze offsets per workout interval using a MyWhoosh workout JSON:

```bash
uv run powermeter-compare \
  --ref-fit data/reference.fit \
  --candidate-fit data/candidate.fit \
  --workout-json data/workout.json \
  --ftp 250
```

### Custom Labels

Override auto-detected power meter names:

```bash
uv run powermeter-compare \
  --ref-fit data/reference.fit \
  --candidate-fit data/candidate.fit \
  --label-ref "Quarq DZero" \
  --label-candidate "Favero Assioma"
```

## Analysis Pipeline

1. **Load FIT Files**: Reads power and timestamp data from both FIT files
2. **Time Alignment**: Resamples to common grid and estimates time offset via cross-correlation
3. **Data Merging**: Merges aligned power data with configurable time tolerance
4. **Dropout Detection**: Identifies invalid/missing data regions
5. **Trailing Averages**: Computes 5s, 10s, and 30s rolling averages
6. **Interval Analysis** (optional): Computes per-interval offsets from workout definition
7. **RANSAC Modeling** (optional): Fits robust linear model of offset vs power
8. **Visualization**: Generates interactive plots for analysis

## Output Plots

### 1. Power Comparison Subplots

Four synchronized subplots showing instantaneous, 5s, 10s, and 30s averaged power:

- Reference power (blue) vs Candidate power (red)
- Gray background indicates dropout regions
- Colored backgrounds show workout intervals (if provided)
- Each subplot annotated with offset statistics

### 2. Offset vs Target Power

Scatter plot showing power offset at different target power levels:

- Blue markers with error bars for each interval
- Green dashed line shows RANSAC regression fit
- Identifies systematic calibration errors

### 3. RANSAC Fit (Detailed)

Enhanced offset vs power plot distinguishing:

- Blue markers: Inlier intervals
- Red X markers: Outlier intervals
- Green dashed line: RANSAC fitted model

### 4. Cross-Correlation Diagnostics

Shows correlation coefficient vs time lag:

- Identifies optimal time offset
- Visualizes alignment quality

## Command-Line Options

```
--ref-fit PATH              Path to reference FIT file (required)
--candidate-fit PATH        Path to candidate FIT file (required)
--label-ref TEXT            Reference power meter label (auto-detected)
--label-candidate TEXT      Candidate power meter label (auto-detected)
--workout-json PATH         MyWhoosh workout JSON for interval analysis
--ftp FLOAT                 Functional Threshold Power (required with --workout-json)
--rampup-exclusion FLOAT    Seconds to exclude from interval start (default: 5.0)
--min-power-threshold FLOAT Minimum power for analysis in watts (default: 50.0)
--time-tolerance TEXT       Time tolerance for alignment (default: "2s")
--max-time-offset INT       Max time offset search range (default: 60)
```

## Use Cases

### Validating a New Power Meter

Compare a new power meter against a trusted reference to verify calibration and accuracy across different power levels.

### Detecting Calibration Drift

Use RANSAC analysis to identify if offset varies with power (e.g., 2% error indicating systematic miscalibration).

### Interval Training Analysis

Analyze per-interval offsets during structured workouts to understand power meter behavior at different intensities.

### Quality Assessment

Identify dropout regions, data gaps, and anomalous intervals that may indicate connectivity or sensor issues.

## RANSAC Output Interpretation

The RANSAC linear model fits: `offset = slope � power + intercept`

- **Slope H 0, Intercept ` 0**: Constant offset (e.g., +8W across all power levels)
- **Slope ` 0, Intercept H 0**: Proportional error (e.g., 3% calibration error)
- **Slope ` 0, Intercept ` 0**: Combined offset and scaling error
- **Outliers**: Intervals with unusual behavior (check for dropouts or anomalies)

## Requirements

- Python e 3.11
- FIT files from compatible cycling power meters
- Optional: MyWhoosh workout JSON files for interval analysis

## Development

```bash
# Run tests
make tests

# Format and lint
make format-lint

# Run with debugging
uv run powermeter-compare --ref-fit data/MyWhoosh.fit --candidate-fit data/Favero.fit --workout-json data/MyWhooshWorkout.json --ftp 320
```

## License

MIT

## Contributing

Contributions welcome! Please ensure:

- All tests pass: `make tests`
- Code is formatted: `make format-lint`
- New features include tests
