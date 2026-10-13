"""Cost-sensitivity analysis on previously frozen, untouched ULB fraud predictions.

Do not optimize thresholds, choose models, or change hyperparameters on the test
labels. All costs are illustrative USD-equivalent *hypotheses*, NOT real bank
loss, chargeback size, realized savings or a deployed policy.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = {
    "review_low_miss_cost": {"false_alert_cost": 20, "missed_fraud_cost": 10},
    "balanced_sensitivity": {"false_alert_cost": 10, "missed_fraud_cost": 100},
    "expensive_missed_fraud": {"false_alert_cost": 10, "missed_fraud_cost": 1000},
}


def evaluate_fixed_thresholds(predictions_csv, report_json, output_json):
    predictions_csv = Path(predictions_csv)
    report_json = Path(report_json)
    output_json = Path(output_json)
    report = json.loads(report_json.read_text(encoding="utf-8"))
    frame = pd.read_csv(predictions_csv)
    needed = {"original_openml_row_position", "original_fraud_label"}
    needed |= {"probability_" + name for name in report["models"]}
    if not needed.issubset(frame.columns):
        raise ValueError("Missing original untouched ULB labels or model probabilities")
    if report["data_contract"]["original_rows"] != 284807:
        raise ValueError("Not the expected verified historic ULB benchmark")
    if len(frame) != report["split"]["test_rows"]:
        raise ValueError("Holdout row count differs from validated original split")
    if (frame["original_openml_row_position"].isna().any()
            or not frame["original_openml_row_position"].is_unique):
        raise ValueError("Missing or duplicated original source row identifiers")
    labels = pd.to_numeric(frame["original_fraud_label"], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all() or labels.sum() != report["split"]["test_fraud"]:
        raise ValueError("Fraud class or total differs from original heldout report")
    if report["selected_model"] not in report["models"]:
        raise ValueError("Missing validation-selected model")

    checks = {}
    for name, prior in report["models"].items():
        threshold = prior["threshold_selected_on_validation_only"]
        if not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid frozen validation threshold")
        probs = pd.to_numeric(frame["probability_" + name], errors="raise").to_numpy()
        if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
            raise ValueError("Invalid test probabilities")
        predicted = (probs >= threshold).astype(int)
        tn = int(np.sum((labels == 0) & (predicted == 0)))
        fp = int(np.sum((labels == 0) & (predicted == 1)))
        fn = int(np.sum((labels == 1) & (predicted == 0)))
        tp = int(np.sum((labels == 1) & (predicted == 1)))
        actual_cm = [[tn, fp], [fn, tp]]
        if actual_cm != prior["untouched_test"]["confusion_matrix_true_0_1"]:
            raise ValueError("Frozen threshold predictions disagree with original audit")
        checks[name] = {
            "validation_frozen_threshold": threshold,
            "tn_fp_fn_tp": [tn, fp, fn, tp],
        }

    selected = report["selected_model"]
    tn, fp, fn, tp = checks[selected]["tn_fp_fn_tp"]
    n_fraud = int(labels.sum())
    decision_tables = {}
    for title, costs in SCENARIOS.items():
        review = costs["false_alert_cost"]
        missed = costs["missed_fraud_cost"]
        hypothetical_baseline = n_fraud * missed
        hypothetical_alerts = fp * review + fn * missed
        net = hypothetical_baseline - hypothetical_alerts
        decision_tables[title] = {
            "assumed_cost_false_positive_alert": review,
            "assumed_cost_missed_fraud": missed,
            "never_flag_anything_hypothetical_cost": hypothetical_baseline,
            "validation_threshold_alert_policy_hypothetical_cost": hypothetical_alerts,
            "illustrative_difference_vs_never_flagging": net,
            "policy_under_hypothetical_costs": "lower_cost" if net > 0 else
                ("equal_cost" if net == 0 else "higher_cost"),
        }
    evidence = {
        "provenance": "Original ULB/Worldline OpenML 1597 public 2013 dataset, frozen heldout results",
        "prediction_csv_sha256": sha256(predictions_csv.read_bytes()).hexdigest(),
        "original_report_sha256": sha256(report_json.read_bytes()).hexdigest(),
        "selected_model": selected,
        "selected_by": "validation average precision only, not these post-hoc costs",
        "n_test": int(len(frame)),
        "actual_fraud_labels": n_fraud,
        "actual_selected_threshold_tn_fp_fn_tp": [tn, fp, fn, tp],
        "all_models_original_confusion_verified": checks,
        "assumptions_not_observed_loss_or_operational_savings": True,
        "threshold_or_model_tuned_using_holdout_labels": False,
        "hypothetical_cost_scenarios": decision_tables,
        "notes": (
            "Illustrates cost sensitivity ONLY on the original already-scored historical "
            "test set, using fixed validation-selected model and threshold. "
            "The alert-review expense and a missed fraud event's cost are invented "
            "hypotheses, NOT observed financial loss. Does not measure delayed drift, "
            "customer harm, chargeback rates, return on investment or a real deployed policy."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items()
                      if k != "all_models_original_confusion_verified"}, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="fraud-detection-finance/results/ulb_real_fraud/heldout_real_ulb_probabilities.csv")
    parser.add_argument("--report", default="fraud-detection-finance/results/ulb_real_fraud/verified_results.json")
    parser.add_argument("--output", default="fraud-detection-finance/results/ulb_real_fraud/frozen_threshold_cost_sensitivity.json")
    args = parser.parse_args()
    evaluate_fixed_thresholds(args.predictions, args.report, args.output)


if __name__ == "__main__":
    main()
