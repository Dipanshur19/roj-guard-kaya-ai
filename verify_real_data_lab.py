"""Quick CLI verification for ROJ Guard Real Data Lab.

Usage:
    python verify_real_data_lab.py "Procurement KPI Analysis Dataset.csv"

This does not modify roj_guard.db or baseline model artifacts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

from real_data_lab import (
    suggest_column_mapping, normalize_dataset, validate_dataset,
    build_supplier_intelligence, train_procurement_lead_time_model,
)


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return 2
    if p.suffix.lower() == ".csv":
        raw = pd.read_csv(p)
    else:
        raw = pd.read_excel(p)
    mapping, confidence = suggest_column_mapping(raw.columns)
    print("\nAuto-mapping")
    for field, source in mapping.items():
        if source:
            print(f"  {field:28s} <- {source:24s} ({confidence[field]*100:.0f}%)")
    normalized = normalize_dataset(raw, mapping)
    report = validate_dataset(normalized)
    print("\nValidation")
    print(f"  mode: {report.mode}")
    print(f"  rows: {report.rows}")
    print(f"  clean delivered: {report.clean_delivered_rows}")
    print(f"  suppliers: {report.vendors}")
    print(f"  categories: {report.categories}")
    for w in report.warnings:
        print(f"  WARNING: {w}")
    for e in report.errors:
        print(f"  ERROR: {e}")
    if report.errors:
        return 1
    if report.mode == "procurement_history":
        kpis = build_supplier_intelligence(normalized)
        print("\nSupplier intelligence")
        if not kpis.empty:
            print(kpis[["Supplier", "Orders", "Clean delivered", "Avg lead time days", "Supplier health score"]].to_string(index=False))
        trained = train_procurement_lead_time_model(normalized)
        print("\nTraining")
        for k, v in trained.items():
            if k != "bundle":
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Procurement KPI Analysis Dataset.csv"))
