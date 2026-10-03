#!/usr/bin/env python3
"""Recompute the exact measured-channel oracle columns without refitting models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/processed/full_race_residual_maxfit_5s.csv"
SUMMARY = ROOT / "reports/residual_method_sweep_summary.json"
OUT = ROOT / "reports/residual_measurement_oracle_audit.json"


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    e = np.asarray(y, float) - np.asarray(p, float)
    return {"n": int(len(e)), "mae_w": float(np.mean(np.abs(e))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "rmse_w": float(np.sqrt(np.mean(e**2))), "bias_w": float(np.mean(e))}


def main() -> None:
    frame = pd.read_csv(CSV, low_memory=False)
    correction = frame.measured_oracle_prediction_w - frame.predicted_w
    corrected = frame.measured_w - (frame.predicted_w + correction)
    frame["oracle_residual_correction_w"] = correction
    frame["oracle_corrected_error_w"] = corrected
    frame.to_csv(CSV, index=False, encoding="utf-8-sig")
    result = {
        "definition": "measured_w = Solar_Watt + Battery_Watt; the oracle reads these contemporaneous target components",
        "correction_matches_observed_residual": metric(frame.target_residual_w.to_numpy(float), correction.to_numpy(float)),
        "corrected_output_error": metric(np.zeros(len(frame)), corrected.to_numpy(float)),
        "valid_as_a_future_predictor": False,
        "interpretation": "This is an arithmetic leakage ceiling, not a residual-learning model or an MPC forecast.",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["measurement_oracle_not_a_forecast"] = result
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
