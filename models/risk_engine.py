"""Transparent, synthetic cardiometabolic indicator for the demo twin.

This score is an educational simulation, not a validated clinical prediction.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class RiskAssessment:
    score: int
    category: str
    factors: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class DigitalTwinState:
    timestamp: str
    score: int
    category: str
    glucose_mg_dl: float
    heart_rate_bpm: float
    systolic_bp_mmhg: float
    steps: int
    sleep_hours: float
    bmi: float
    hba1c_pct: float
    activity_days_per_week: int
    diet_adherence: str
    carbohydrate_reduction_percent: int = 0
    medication_adherence: str = "Not applicable"


@dataclass(frozen=True)
class ScenarioResult:
    current_state: DigitalTwinState
    simulated_state: DigitalTwinState
    current_assessment: RiskAssessment
    factor_deltas: tuple[tuple[str, int], ...]


RISK_FACTOR_GUIDANCE = (
    ("Age", "Adds up to 8 points, increasing gradually from age 35 and capped at age 75."),
    ("Baseline HbA1c", "<5.7%: 0; 5.7-5.9%: 12; 6.0-6.4%: 18; >=6.5%: 25 points."),
    ("BMI", "<25: 0; 25-29.9: 6; >=30 kg/m²: 12 points."),
    ("Baseline systolic BP", "<120: 0; 120-129: 2; 130-139: 4; >=140 mmHg: 6 points."),
    ("Current systolic BP", "Uses the same illustrative 0/2/4/6-point systolic ranges as baseline BP."),
    ("Baseline total cholesterol", "<200: 0; 200-239: 4; >=240 mg/dL: 8 points."),
    ("Family history", "5 points when the synthetic profile flags a family history of type 2 diabetes."),
    ("Daily steps", "Up to 8 points, tapering linearly to 0 at 10,000 steps/day."),
    ("Sleep duration", "7-9 hours: 0; 6-<7 hours: 2; <6 or >9 hours: 5 points."),
    ("Current glucose", "<140: 0; 140-179: 5; >=180 mg/dL: 12 points."),
    ("Resting heart rate", "<=80: 0; 81-95: 2; >95 bpm: 5 points."),
)


def _value(record: Mapping[str, object], key: str) -> float:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid numeric value: {key}") from error
    if not isfinite(value):
        raise ValueError(f"Value must be finite: {key}")
    return value


def _category(score: int) -> str:
    if score < 25:
        return "Lower indicator"
    if score < 50:
        return "Moderate indicator"
    if score < 75:
        return "Elevated indicator"
    return "Higher indicator"


def assess_risk(
    profile: Mapping[str, object], observation: Mapping[str, object]
) -> RiskAssessment:
    """Combine baseline EHR and a current simulated observation into 0-100 points."""
    age = _value(profile, "age")
    hba1c = _value(profile, "hba1c_pct")
    bmi = _value(profile, "bmi")
    baseline_systolic = _value(profile, "systolic_bp_mmhg")
    current_systolic = _value(observation, "systolic_bp_mmhg")
    cholesterol = _value(profile, "total_cholesterol_mg_dl")
    family_history = _value(profile, "family_history_t2d")
    steps = _value(observation, "steps")
    sleep = _value(observation, "sleep_hours")
    glucose = _value(observation, "glucose_mg_dl")
    heart_rate = _value(observation, "heart_rate_bpm")

    factors = (
        ("Age", round(min(max(age - 35, 0) / 40, 1) * 8)),
        (
            "Baseline HbA1c",
            0 if hba1c < 5.7 else 12 if hba1c < 6.0 else 18 if hba1c < 6.5 else 25,
        ),
        ("BMI", 0 if bmi < 25 else 6 if bmi < 30 else 12),
        (
            "Baseline systolic BP",
            0 if baseline_systolic < 120 else 2 if baseline_systolic < 130 else 4 if baseline_systolic < 140 else 6,
        ),
        (
            "Current systolic BP",
            0 if current_systolic < 120 else 2 if current_systolic < 130 else 4 if current_systolic < 140 else 6,
        ),
        (
            "Baseline total cholesterol",
            0 if cholesterol < 200 else 4 if cholesterol < 240 else 8,
        ),
        ("Family history", 5 if family_history >= 1 else 0),
        ("Daily steps", round(max(0, min(8, (10000 - steps) * 8 / 10000)))),
        ("Sleep duration", 0 if 7 <= sleep <= 9 else 2 if 6 <= sleep < 7 else 5),
        ("Current glucose", 0 if glucose < 140 else 5 if glucose < 180 else 12),
        ("Resting heart rate", 0 if heart_rate <= 80 else 2 if heart_rate <= 95 else 5),
    )
    score = min(100, max(0, round(sum(points for _, points in factors))))
    return RiskAssessment(score=score, category=_category(score), factors=factors)


def assess_step_scenario(
    profile: Mapping[str, object],
    observation: Mapping[str, object],
    increase_percent: int,
) -> RiskAssessment:
    """Re-score with a hypothetical activity increase, capped at 30,000 steps."""
    if not 0 <= increase_percent <= 100:
        raise ValueError("increase_percent must be between 0 and 100")
    scenario = dict(observation)
    scenario["steps"] = min(
        30000, _value(observation, "steps") * (1 + increase_percent / 100)
    )
    return assess_risk(profile, scenario)


def calculate_twin_state(
    profile: Mapping[str, object],
    observation: Mapping[str, object],
    activity_days_per_week: int = 2,
    diet_adherence: str = "Moderate",
    carbohydrate_reduction_percent: int = 0,
    medication_adherence: str = "Not applicable",
) -> DigitalTwinState:
    """Create a current synthetic twin snapshot from the existing EHR and stream."""
    assessment = assess_risk(profile, observation)
    return DigitalTwinState(
        timestamp=str(observation["timestamp"]),
        score=assessment.score,
        category=assessment.category,
        glucose_mg_dl=_value(observation, "glucose_mg_dl"),
        heart_rate_bpm=_value(observation, "heart_rate_bpm"),
        systolic_bp_mmhg=_value(observation, "systolic_bp_mmhg"),
        steps=int(_value(observation, "steps")),
        sleep_hours=_value(observation, "sleep_hours"),
        bmi=_value(profile, "bmi"),
        hba1c_pct=_value(profile, "hba1c_pct"),
        activity_days_per_week=activity_days_per_week,
        diet_adherence=diet_adherence,
        carbohydrate_reduction_percent=carbohydrate_reduction_percent,
        medication_adherence=medication_adherence,
    )


def get_risk_contributors(
    assessment: RiskAssessment, limit: int = 5
) -> tuple[tuple[str, int], ...]:
    """Return the largest positive contributors already present in the score."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return tuple(
        sorted(
            ((name, points) for name, points in assessment.factors if points > 0),
            key=lambda factor: factor[1],
            reverse=True,
        )[:limit]
    )


def calculate_simulated_state(
    profile: Mapping[str, object],
    observation: Mapping[str, object],
    *,
    steps: int,
    sleep_hours: float,
    activity_days_per_week: int,
    diet_adherence: str,
    carbohydrate_reduction_percent: int = 0,
    medication_adherence: str = "Not applicable",
) -> ScenarioResult:
    """Apply explicit demonstration-only rules to a hypothetical twin scenario.

    Activity frequency and diet have neutral baselines of 2 days/week and
    Moderate adherence. These are scenario defaults, not patient-reported data.
    No modeled effect is applied to glucose, heart rate, or blood pressure.
    """
    steps_value = _value({"steps": steps}, "steps")
    sleep_value = _value({"sleep_hours": sleep_hours}, "sleep_hours")
    if not 0 <= steps_value <= 30000:
        raise ValueError("steps must be between 0 and 30000")
    if not 0 <= sleep_value <= 24:
        raise ValueError("sleep_hours must be between 0 and 24")
    if not isinstance(activity_days_per_week, int) or not 0 <= activity_days_per_week <= 7:
        raise ValueError("activity_days_per_week must be an integer from 0 to 7")

    diet_points = {"Low": 2, "Moderate": 0, "High": -2}
    if diet_adherence not in diet_points:
        raise ValueError("diet_adherence must be Low, Moderate, or High")
    if not isinstance(carbohydrate_reduction_percent, int) or not 0 <= carbohydrate_reduction_percent <= 50:
        raise ValueError("carbohydrate_reduction_percent must be an integer from 0 to 50")
    medication_points = {"Not applicable": 0, "Consistent": -1, "Intermittent": 2}
    if medication_adherence not in medication_points:
        raise ValueError("medication_adherence must be Not applicable, Consistent, or Intermittent")

    current_assessment = assess_risk(profile, observation)
    current_state = calculate_twin_state(profile, observation)
    scenario_observation = dict(observation)
    scenario_observation["steps"] = int(steps_value)
    scenario_observation["sleep_hours"] = sleep_value
    scenario_assessment = assess_risk(profile, scenario_observation)

    current_factors = dict(current_assessment.factors)
    scenario_factors = dict(scenario_assessment.factors)
    activity_delta = max(-5, min(2, 2 - activity_days_per_week))
    carbohydrate_delta = -min(3, carbohydrate_reduction_percent // 15)
    factor_deltas = (
        ("Daily steps", scenario_factors["Daily steps"] - current_factors["Daily steps"]),
        (
            "Sleep duration",
            scenario_factors["Sleep duration"] - current_factors["Sleep duration"],
        ),
        ("Activity frequency", activity_delta),
        ("Diet adherence", diet_points[diet_adherence]),
        ("Carbohydrate reduction", carbohydrate_delta),
        ("Medication adherence", medication_points[medication_adherence]),
    )
    simulated_score = min(
        100,
        max(
            0,
            scenario_assessment.score
            + activity_delta
            + diet_points[diet_adherence]
            + carbohydrate_delta
            + medication_points[medication_adherence],
        ),
    )
    simulated_profile_state = calculate_twin_state(
        profile,
        scenario_observation,
        activity_days_per_week=activity_days_per_week,
        diet_adherence=diet_adherence,
        carbohydrate_reduction_percent=carbohydrate_reduction_percent,
        medication_adherence=medication_adherence,
    )
    simulated_state = DigitalTwinState(
        **{
            **simulated_profile_state.__dict__,
            "score": simulated_score,
            "category": _category(simulated_score),
        }
    )
    return ScenarioResult(
        current_state=current_state,
        simulated_state=simulated_state,
        current_assessment=current_assessment,
        factor_deltas=factor_deltas,
    )


def generate_scenario_explanations(result: ScenarioResult) -> tuple[str, ...]:
    """Explain only the lifestyle inputs changed in the deterministic scenario."""
    current = result.current_state
    simulated = result.simulated_state
    deltas = dict(result.factor_deltas)
    explanations: list[str] = []

    if simulated.steps != current.steps:
        points = abs(deltas["Daily steps"])
        if deltas["Daily steps"] < 0:
            explanations.append(
                f"The higher step target reduces the simulated activity contribution by {points} point(s)."
            )
        elif deltas["Daily steps"] > 0:
            explanations.append(
                f"The lower step target adds {points} point(s) to the simulated activity contribution."
            )
        else:
            explanations.append("The step target changed but stayed within the same rule-based score band.")

    if simulated.sleep_hours != current.sleep_hours:
        points = abs(deltas["Sleep duration"])
        if deltas["Sleep duration"] < 0:
            explanations.append(
                f"The longer sleep setting reduces the simulated sleep contribution by {points} point(s)."
            )
        elif deltas["Sleep duration"] > 0:
            explanations.append(
                f"The shorter sleep setting adds {points} point(s) to the simulated sleep contribution."
            )
        else:
            explanations.append("The sleep setting changed but stayed within the same rule-based score band.")

    if simulated.activity_days_per_week != current.activity_days_per_week:
        if deltas["Activity frequency"] < 0:
            explanations.append("More active days reduce the hypothetical activity-frequency points.")
        elif deltas["Activity frequency"] > 0:
            explanations.append("Fewer active days add to the hypothetical activity-frequency points.")
        else:
            explanations.append("The activity-frequency setting changed within its capped rule-based range.")

    if simulated.diet_adherence != current.diet_adherence:
        if deltas["Diet adherence"] < 0:
            explanations.append("Higher diet-adherence setting lowers the hypothetical lifestyle points.")
        else:
            explanations.append("Lower diet-adherence setting adds to the hypothetical lifestyle points.")

    if simulated.carbohydrate_reduction_percent != current.carbohydrate_reduction_percent:
        if deltas["Carbohydrate reduction"] < 0:
            explanations.append(
                "The hypothetical carbohydrate reduction lowers only the rule-based diet contribution."
            )
        else:
            explanations.append("Carbohydrate intake was not reduced in this scenario.")

    if simulated.medication_adherence != current.medication_adherence:
        if deltas["Medication adherence"] < 0:
            explanations.append(
                "The selected medication-adherence setting lowers the demonstration-only scenario points."
            )
        elif deltas["Medication adherence"] > 0:
            explanations.append(
                "The selected medication-adherence setting adds demonstration-only scenario points."
            )
        else:
            explanations.append("Medication adherence is marked not applicable in this scenario.")

    if not explanations:
        return ("No scenario inputs changed; the simulated twin matches the current baseline.",)
    return tuple(explanations[:4])
