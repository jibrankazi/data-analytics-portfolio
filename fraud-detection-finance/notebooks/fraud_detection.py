"""Fraud detection project's working research entrypoint.

This replaces a broken Windows-only PaySim script that oversampled before
splitting (invalid held-out evaluation). The default research pipeline
uses published ORIGINAL ULB/Worldline anonymized card transaction records
from OpenML id=1597, not PaySim synthetic transactions.

Usage from repository root:
    python fraud-detection-finance/notebooks/fraud_detection.py

This is a historical research classifier, not live financial surveillance.
The experimental implementation is maintained in ../run_real_ulb_fraud.py.
"""
import argparse
from pathlib import Path
import sys

SOURCE_PROJECT = Path(__file__).resolve().parents[1]
if str(SOURCE_PROJECT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PROJECT))

from run_real_ulb_fraud import run as run_real_ulb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", default="fraud-detection-finance/results/ulb_real_fraud",
        help="Directory for original-source validation, held-out metrics and plots",
    )
    args = p.parse_args()
    return run_real_ulb(output=args.output)


if __name__ == "__main__":
    main()
