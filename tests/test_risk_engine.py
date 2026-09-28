import unittest

from backend.simulator import (
    generate_synthetic_history,
    get_synthetic_alerts,
    load_observations,
    load_profiles,
    replay_observation,
)
from models.risk_engine import (
    assess_risk,
    assess_step_scenario,
    calculate_simulated_state,
    generate_scenario_explanations,
    get_risk_contributors,
)


class RiskEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = {row["patient_id"]: row for row in load_profiles()}
        cls.observations = load_observations()

    def test_score_is_bounded_and_factors_sum_to_score(self):
        profile = self.profiles["TT-318"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-318")
        result = assess_risk(profile, sample)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 100)
        self.assertEqual(result.score, sum(points for _, points in result.factors))

    def test_more_steps_never_increases_this_indicator(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        baseline = assess_risk(profile, sample)
        scenario = assess_step_scenario(profile, sample, 20)
        self.assertLess(scenario.score, baseline.score)

    def test_higher_current_systolic_pressure_increases_indicator(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        lower_pressure = dict(sample, systolic_bp_mmhg=115)
        higher_pressure = dict(sample, systolic_bp_mmhg=150)
        self.assertGreater(
            assess_risk(profile, higher_pressure).score,
            assess_risk(profile, lower_pressure).score,
        )

    def test_scenario_rejects_out_of_range_increase(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        with self.assertRaises(ValueError):
            assess_step_scenario(profile, sample, 101)

    def test_replay_wraps_within_selected_patient(self):
        sample, cursor = replay_observation(self.observations, "TT-207", -1)
        self.assertEqual(cursor, 0)
        self.assertEqual(sample["patient_id"], "TT-207")

    def test_scenario_baseline_matches_current_score(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        result = calculate_simulated_state(
            profile,
            sample,
            steps=sample["steps"],
            sleep_hours=sample["sleep_hours"],
            activity_days_per_week=2,
            diet_adherence="Moderate",
        )
        self.assertEqual(result.current_state.score, 59)
        self.assertEqual(result.simulated_state.score, result.current_state.score)

    def test_scenario_is_deterministic_and_explains_changed_inputs(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        arguments = {
            "steps": 3000,
            "sleep_hours": 7.5,
            "activity_days_per_week": 4,
            "diet_adherence": "High",
            "carbohydrate_reduction_percent": 20,
            "medication_adherence": "Consistent",
        }
        result = calculate_simulated_state(profile, sample, **arguments)
        repeated = calculate_simulated_state(profile, sample, **arguments)
        self.assertEqual(result, repeated)
        self.assertLess(result.simulated_state.score, result.current_state.score)
        explanations = generate_scenario_explanations(result)
        self.assertGreaterEqual(len(explanations), 2)
        self.assertLessEqual(len(explanations), 4)

    def test_scenario_rejects_invalid_inputs(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        with self.assertRaises(ValueError):
            calculate_simulated_state(
                profile,
                sample,
                steps=40000,
                sleep_hours=7,
                activity_days_per_week=4,
                diet_adherence="High",
                carbohydrate_reduction_percent=0,
                medication_adherence="Not applicable",
            )

    def test_scenario_rejects_invalid_adherence_options(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        with self.assertRaises(ValueError):
            calculate_simulated_state(
                profile,
                sample,
                steps=2000,
                sleep_hours=7,
                activity_days_per_week=3,
                diet_adherence="High",
                medication_adherence="Unsupervised dose change",
            )

    def test_contributors_return_largest_positive_factors(self):
        profile = self.profiles["TT-104"]
        sample = next(row for row in self.observations if row["patient_id"] == "TT-104")
        contributors = get_risk_contributors(assess_risk(profile, sample), limit=3)
        self.assertEqual(contributors[0][0], "Baseline HbA1c")
        self.assertEqual(len(contributors), 3)

    def test_added_profiles_have_matching_wearable_data_and_distinct_scores(self):
        self.assertIn("TT-531", self.profiles)
        self.assertIn("TT-642", self.profiles)
        sample_by_patient = {
            row["patient_id"]: row
            for row in self.observations
            if row["patient_id"] in {"TT-531", "TT-642"}
        }
        optimal_score = assess_risk(self.profiles["TT-531"], sample_by_patient["TT-531"]).score
        jordan_score = assess_risk(
            self.profiles["TT-104"],
            next(row for row in self.observations if row["patient_id"] == "TT-104"),
        ).score
        diabetes_score = assess_risk(self.profiles["TT-642"], sample_by_patient["TT-642"]).score
        self.assertLess(optimal_score, jordan_score)
        self.assertGreater(diabetes_score, jordan_score)

    def test_longitudinal_history_is_deterministic_and_keeps_original_data(self):
        source_rows = [row for row in self.observations if row["patient_id"] == "TT-104"]
        weekly = generate_synthetic_history(self.observations, "TT-104", 7)
        repeated = generate_synthetic_history(self.observations, "TT-104", 7)
        monthly = generate_synthetic_history(self.observations, "TT-104", 30)
        self.assertEqual(len(weekly), 42)
        self.assertEqual(len(monthly), 180)
        self.assertEqual(weekly, repeated)
        self.assertTrue(all(row["synthetic_history"] for row in monthly))
        self.assertEqual(len(source_rows), 6)

    def test_longitudinal_history_rejects_unsupported_ranges(self):
        with self.assertRaises(ValueError):
            generate_synthetic_history(self.observations, "TT-104", 14)

    def test_alert_feed_uses_selected_patient_synthetic_thresholds(self):
        jordan_alerts = get_synthetic_alerts(self.observations, "TT-104")
        diabetes_alerts = get_synthetic_alerts(self.observations, "TT-318")
        self.assertEqual(jordan_alerts, [])
        self.assertTrue(any("Elevated synthetic glucose" in alert for alert in diabetes_alerts))
        self.assertTrue(any("Elevated synthetic systolic BP" in alert for alert in diabetes_alerts))


if __name__ == "__main__":
    unittest.main()
