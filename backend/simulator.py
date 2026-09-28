"""Load synthetic records and replay wearable observations as a live stream."""

import csv
from datetime import date, datetime, time, timedelta
from math import cos, sin
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_profiles(path: Path = DATA_DIR / "patient_profiles.csv") -> list[dict[str, Any]]:
    """Read baseline EHR profiles, converting numeric fields from CSV strings."""
    profiles: list[dict[str, Any]] = []
    for row in _read_csv(path):
        profile: dict[str, Any] = dict(row)
        for key in (
            "age",
            "bmi",
            "hba1c_pct",
            "total_cholesterol_mg_dl",
            "systolic_bp_mmhg",
            "family_history_t2d",
        ):
            profile[key] = float(row[key])
        profile["age"] = int(profile["age"])
        profile["family_history_t2d"] = int(profile["family_history_t2d"])
        profiles.append(profile)
    return profiles


def load_observations(
    path: Path = DATA_DIR / "wearable_timeseries.csv",
) -> list[dict[str, Any]]:
    """Read chronologically ordered synthetic wearable and home readings."""
    observations: list[dict[str, Any]] = []
    for row in _read_csv(path):
        observation: dict[str, Any] = dict(row)
        for key in (
            "heart_rate_bpm",
            "steps",
            "sleep_hours",
            "glucose_mg_dl",
            "systolic_bp_mmhg",
        ):
            observation[key] = float(row[key])
        observation["steps"] = int(observation["steps"])
        observations.append(observation)
    observations.sort(key=lambda item: (item["patient_id"], item["timestamp"]))
    return observations


def replay_observation(
    observations: list[dict[str, Any]], patient_id: str, cursor: int
) -> tuple[dict[str, Any], int]:
    """Return the next observation for a patient and its wrapped replay cursor."""
    patient_observations = [
        row for row in observations if row["patient_id"] == patient_id
    ]
    if not patient_observations:
        raise ValueError(f"No wearable observations for patient {patient_id}")
    next_cursor = (cursor + 1) % len(patient_observations)
    return dict(patient_observations[next_cursor]), next_cursor


def generate_synthetic_history(
    observations: list[dict[str, Any]], patient_id: str, days: int
) -> list[dict[str, Any]]:
    """Create a deterministic 7- or 30-day synthetic extension of a patient's stream.

    These generated points are demo history, not additional measured records.
    Each day reuses that patient's six supplied time-of-day patterns with small,
    deterministic variations so longitudinal charts never imply real collection.
    """
    if days not in {7, 30}:
        raise ValueError("days must be 7 or 30")
    patient_rows = [
        row for row in observations if row["patient_id"] == patient_id
    ]
    if not patient_rows:
        raise ValueError(f"No wearable observations for patient {patient_id}")

    templates: dict[int, dict[str, Any]] = {}
    for row in patient_rows:
        sample_time = datetime.fromisoformat(row["timestamp"])
        templates[sample_time.hour] = row
    if not templates:
        raise ValueError(f"No timestamped readings for patient {patient_id}")

    reference_date = max(
        date.fromisoformat(row["timestamp"][:10]) for row in patient_rows
    )
    patient_code = sum(ord(character) for character in patient_id)
    history: list[dict[str, Any]] = []
    hours = sorted(templates)

    for day_index in range(days):
        sample_date = reference_date - timedelta(days=days - day_index - 1)
        phase = (day_index + patient_code % 11) * 0.47
        glucose_offset = 8 * sin(phase)
        heart_rate_offset = 2 * sin(phase + 0.8)
        pressure_offset = 3 * cos(phase + 0.3)
        sleep_offset = 0.25 * cos(phase)
        step_multiplier = 1 + 0.08 * sin(phase + 1.2)

        for hour in hours:
            template = templates[hour]
            sample = dict(template)
            sample["timestamp"] = datetime.combine(sample_date, time(hour)).isoformat()
            sample["glucose_mg_dl"] = round(
                max(40, min(350, float(template["glucose_mg_dl"]) + glucose_offset)), 1
            )
            sample["heart_rate_bpm"] = round(
                max(35, float(template["heart_rate_bpm"]) + heart_rate_offset), 1
            )
            sample["systolic_bp_mmhg"] = round(
                max(70, float(template["systolic_bp_mmhg"]) + pressure_offset), 1
            )
            sample["sleep_hours"] = round(
                max(0, min(24, float(template["sleep_hours"]) + sleep_offset)), 1
            )
            sample["steps"] = max(0, round(int(template["steps"]) * step_multiplier))
            sample["synthetic_history"] = True
            history.append(sample)

    history.sort(key=lambda row: row["timestamp"])
    return history


def get_synthetic_alerts(
    observations: list[dict[str, Any]], patient_id: str, limit: int = 5
) -> list[str]:
    """Summarize recent demo-only glucose rises and systolic threshold readings."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    patient_rows = sorted(
        (row for row in observations if row["patient_id"] == patient_id),
        key=lambda row: row["timestamp"],
    )
    alerts: list[str] = []
    previous: dict[str, Any] | None = None
    for row in patient_rows:
        sample_time = datetime.fromisoformat(row["timestamp"]).strftime("%H:%M")
        glucose = float(row["glucose_mg_dl"])
        pressure = float(row["systolic_bp_mmhg"])
        previous_glucose = float(previous["glucose_mg_dl"]) if previous else None

        if glucose >= 180:
            alerts.append(
                f"Elevated synthetic glucose at {sample_time}: {glucose:.0f} mg/dL "
                "(illustrative alert threshold: 180 mg/dL)."
            )
        elif previous_glucose is not None and glucose - previous_glucose >= 40:
            alerts.append(
                f"Synthetic glucose spike at {sample_time}: increased "
                f"{glucose - previous_glucose:.0f} mg/dL between samples."
            )

        if pressure >= 140:
            alerts.append(
                f"Elevated synthetic systolic BP at {sample_time}: {pressure:.0f} mmHg "
                "(illustrative alert threshold: 140 mmHg)."
            )
        previous = row
    return alerts[-limit:]
