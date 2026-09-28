# Twin Health: Cardiometabolic Digital Twin PoC

A local, interactive proof of concept for fusing synthetic EHR baselines with replayed wearable and home-monitoring observations. The app surfaces a Digital Twin State, an interpretable cardiometabolic indicator, sample-level alerts, 24-hour and generated longitudinal trends, and a multi-input scenario builder.

> **Clinical safety:** This is a hackathon demonstration, not a medical device or clinical decision-support system. Its score is a hand-authored simulation indicator, not a probability or validated prediction. It must not be used to diagnose, treat, or triage anyone. Every patient and sensor record in this repository is synthetic.

## Architecture

```text
data/patient_profiles.csv ─┐
                           ├─ backend/simulator.py ─ models/risk_engine.py ─ dashboard/app.py
data/wearable_timeseries.csv┘       CSV loading             rule score          Streamlit UI
```

- **Static EHR inputs:** six synthetic personas spanning lower-indicator, intermediate, and type 2 diabetes examples; fields include age, sex, BMI, baseline HbA1c, total cholesterol, systolic blood pressure, family history, and medical-history label.
- **Dynamic inputs:** timestamped heart rate, daily step count, sleep duration, glucose, and home systolic blood pressure samples. The dashboard replays these rows as a simulated stream; it does not connect to a device or external service.
- **Fusion and scoring:** `models/risk_engine.py` combines baseline and current signals into a bounded 0-100 point indicator. Each factor's contribution is shown in the dashboard. Thresholds and weights are illustrative and are not clinically validated.
- **What-if simulation:** daily steps, sleep hours, active days per week, diet-adherence level, hypothetical carbohydrate reduction, and medication-adherence scenario feed a deterministic transparent formula. It is not an estimate of a treatment effect.
- **Longitudinal trends:** 7- and 30-day views are deterministic extensions of the supplied synthetic reading patterns. They are generated examples, not additional measurements.
- **UI:** patient selection, replay controls, current twin metrics, an illustrative glucose alert, trends, factor contribution guidance, and current-versus-simulated comparison with generated explanations.
- **Patient summary export:** the Live overview downloads a CSV containing the selected synthetic baseline EHR, latest replayed vitals, and current rule-based indicator.
- **Synthetic alert feed:** Live overview summarizes glucose rises of at least 40 mg/dL between samples, glucose readings at or above 180 mg/dL, and systolic readings at or above 140 mmHg. These are illustrative demo thresholds, not clinical alerts.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). On macOS/Linux, activate the environment with `source .venv/bin/activate`.

Use the sidebar to choose a synthetic profile. Click **Next reading** to replay the next sample, or enable **Auto-advance stream** for five-second updates. The bundled sample sequence wraps when it reaches the end. In **Trends**, choose 24 hours, 7 days, or 30 days; the longer views are labeled synthetic extensions. In **What-if**, change lifestyle assumptions and select **Run Scenario** to compare the captured current state with the deterministic simulated state.

## Repository map

```text
data/
  patient_profiles.csv       Synthetic baseline EHR records
  wearable_timeseries.csv    Synthetic time-series observations
models/
  risk_engine.py             Pure-Python scoring and scenario logic
backend/
  simulator.py               CSV ingestion and deterministic stream replay
dashboard/
  app.py                     Streamlit clinician/patient dashboard
tests/
  test_risk_engine.py        Standard-library unit tests
requirements.txt             Dashboard dependencies
```

## Data dictionary

`patient_profiles.csv` has one baseline row per synthetic patient. `patient_id` joins it to `wearable_timeseries.csv`; all values are fictional. The time-series file is intentionally small and repeatable for a demo. Fields use mg/dL for glucose and total cholesterol, percent for HbA1c and sleep hours, kg/m² for BMI, mmHg for systolic BP, bpm for heart rate, and steps per day for activity. Timestamps are sample labels for replay, not a real-time device clock.

## Model details and limitations

The current score adds bounded point contributions from age (up to 8), HbA1c (25), BMI (12), baseline systolic BP (6), current systolic BP (6), cholesterol (8), family history (5), steps (up to 8, tapering to zero at 10,000 steps), sleep (5), current glucose (12), and heart rate (5). Category bands are `<25`, `25-49`, `50-74`, and `75-100`. A glucose reading at or above 180 mg/dL triggers a demonstration alert. Exact illustrative thresholds are available in the Trends view.

Scenario defaults are 2 active days/week, Moderate diet adherence, 0% carbohydrate reduction, and medication adherence Not applicable; these are demo assumptions and are not stored patient history. Scenario adjustments are: activity frequency `clamp(2 - days, -5, 2)`, Low/Moderate/High diet `+2/0/-2`, carbohydrate reduction `-floor(percent / 15)` capped at `-3`, and Not applicable/Consistent/Intermittent medication adherence `0/-1/+2`. Step and sleep adjustments reuse the existing risk-engine contribution rules. All scenario changes are hypothetical; glucose, blood pressure, BMI, and HbA1c are held constant because this prototype has no validated model for forecasting them.

The 7-/30-day history helper repeats each patient's supplied time-of-day pattern with deterministic, small signal variations. It exists to demonstrate longitudinal visualization only; it is not measured history and should never be presented as such.

There is no trained machine-learning model, calibration, outcome label, clinical validation, real patient data, authentication, persistence, device integration, or production security layer. Before any real-world use, involve qualified clinicians and privacy/security experts; establish lawful data governance, representative cohorts, prospective validation, monitoring, accessibility, and applicable regulatory review.

## Tests

The scoring and replay tests use only the Python standard library:

```powershell
python -m unittest discover -s tests -v
```
