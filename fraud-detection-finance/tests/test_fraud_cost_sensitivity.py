"""Offline-only verification of cost sensitivity; never claim toy costs are bank losses."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "analyze_fraud_costs.py"
SPEC = importlib.util.spec_from_file_location("fraud_cost_analysis", SRC)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FrozenCostsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.probs = base / "heldout.csv"
        self.report = base / "report.json"
        self.out = base / "costs.json"
        pd.DataFrame({
            "original_openml_row_position": range(8),
            "original_fraud_label": [1, 0, 1, 0, 0, 1, 0, 0],
            "probability_demo": [.9, .8, .2, .1, .0, .7, .1, .4],
        }).to_csv(self.probs, index=False)
        self.document = {
            "data_contract": {"original_rows": 284807},
            "split": {"test_rows": 8, "test_fraud": 3},
            "selected_model": "demo",
            "models": {
                "demo": {
                    "threshold_selected_on_validation_only": .5,
                    "untouched_test": {
                        "confusion_matrix_true_0_1": [[4, 1], [1, 2]],
                    },
                },
            },
        }
        self.save()

    def save(self):
        self.report.write_text(json.dumps(self.document))

    def test_independently_recomputes_original_confusion_and_scenario_math(self):
        result = MOD.evaluate_fixed_thresholds(self.probs, self.report, self.out)
        self.assertEqual(result["actual_selected_threshold_tn_fp_fn_tp"], [4, 1, 1, 2])
        scenarios = result["hypothetical_cost_scenarios"]
        self.assertEqual(scenarios["review_low_miss_cost"]["illustrative_difference_vs_never_flagging"], 0)
        self.assertEqual(scenarios["balanced_sensitivity"]["illustrative_difference_vs_never_flagging"], 190)
        self.assertEqual(scenarios["expensive_missed_fraud"]["illustrative_difference_vs_never_flagging"], 1990)
        self.assertFalse(result["threshold_or_model_tuned_using_holdout_labels"])
        self.assertTrue(result["assumptions_not_observed_loss_or_operational_savings"])
        self.assertEqual(json.loads(self.out.read_text()), result)
        self.assertEqual(MOD.evaluate_fixed_thresholds(self.probs, self.report, self.out), result)

    def test_rejects_original_confusion_tampering(self):
        self.document["models"]["demo"]["untouched_test"]["confusion_matrix_true_0_1"] = [[3, 2], [1, 2]]
        self.save()
        with self.assertRaisesRegex(ValueError, "disagree"):
            MOD.evaluate_fixed_thresholds(self.probs, self.report, self.out)
        self.assertFalse(self.out.exists())

    def test_rejects_bad_probabilities_and_label_tampering(self):
        sample = pd.read_csv(self.probs)
        sample.loc[0, "probability_demo"] = 1.5
        sample.to_csv(self.probs, index=False)
        with self.assertRaisesRegex(ValueError, "probabilities"):
            MOD.evaluate_fixed_thresholds(self.probs, self.report, self.out)
        sample.loc[0, "probability_demo"] = .9
        sample.loc[0, "original_fraud_label"] = 0
        sample.to_csv(self.probs, index=False)
        with self.assertRaisesRegex(ValueError, "Fraud class"):
            MOD.evaluate_fixed_thresholds(self.probs, self.report, self.out)


if __name__ == "__main__":
    unittest.main()
