#!/usr/bin/env python3
"""Run the existing nested online evaluator with route distance as a feature."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).with_name("evaluate_online_kernel_residual.py")
SPEC = importlib.util.spec_from_file_location("online_kernel_distance_base", BASE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load evaluator at {BASE}")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)

evaluator.FEATURES = [*evaluator.FEATURES, "distance_m"]
evaluator.OUT_BLOCKS = ROOT / "reports/residual_online_kernel_distance_oof.csv"
evaluator.OUT_DAYS = ROOT / "reports/residual_online_kernel_distance_by_day.csv"
evaluator.OUT_SUMMARY = ROOT / "reports/residual_online_kernel_distance_summary.json"
evaluator.OUT_REPORT = ROOT / "reports/residual_online_kernel_distance_report.md"
evaluator.OUT_MODEL = ROOT / "models/vehicle_residual_online_kernel_distance_candidate.json"
evaluator.ROWS_IN = ROOT / "data/processed/full_race_residual_corrected_5s.csv"
evaluator.ROWS_OUT = ROOT / "data/processed/full_race_residual_online_kernel_distance_5s.csv"

if __name__ == "__main__":
    evaluator.main()
