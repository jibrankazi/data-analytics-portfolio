"""Independent real City of Toronto 311 dataset findings, no fabricated cases.

This dataset includes requests through several channels. It is NOT call-center
call volume and cannot independently establish required agent staffing.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

SOURCE = "https://open.toronto.ca/dataset/311-service-requests-customer-initiated/"
DATA = "toronto-311-analysis/data/SR2025.csv"

def evaluate(path=DATA):
    d = pd.read_csv(path, encoding="latin1", low_memory=False, on_bad_lines="error")
    required = {"Creation Date", "Service Request Type", "Status"}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"Expected City of Toronto service-request columns missing: {sorted(missing)}")
    date = pd.to_datetime(d["Creation Date"], errors="coerce")
    valid = d.loc[date.notna()].copy()
    valid["created_at"] = date[date.notna()]
    if len(valid) < 10000:
        raise ValueError("Far too few real observations to reproduce the 2025 annual analysis")
    period = valid[valid.created_at.dt.year.eq(2025)].copy()
    if period.empty:
        raise ValueError("No records from calendar 2025; refusing fabricated fallback")
    months = period.groupby(period.created_at.dt.month).size()
    days = period.groupby(period.created_at.dt.strftime("%Y-%m-%d")).size()
    types = period["Service Request Type"].value_counts(dropna=True)
    statuses = period["Status"].value_counts(dropna=True)
    return {
        "source": SOURCE,
        "file": str(path),
        "dataset_rows": int(len(d)),
        "valid_date_rows": int(len(valid)),
        "calendar_2025_requests": int(len(period)),
        "distinct_request_types": int(types.size),
        "first_observed": str(period.created_at.min()),
        "last_observed": str(period.created_at.max()),
        "peak_month": int(months.idxmax()),
        "peak_month_requests": int(months.max()),
        "peak_day": str(days.idxmax()),
        "peak_day_requests": int(days.max()),
        "most_frequent_request_type": str(types.index[0]),
        "most_frequent_request_count": int(types.iloc[0]),
        "top_statuses": {str(k): int(v) for k,v in statuses.head(5).items()},
        "channel_caveat": "All 311 service request channels combined; cannot infer phone call volumes or staffing",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default=DATA)
    p.add_argument("--output",default="toronto-311-analysis/results/verified_real_2025.json")
    args=p.parse_args()
    result=evaluate(args.input)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
