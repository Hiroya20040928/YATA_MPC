#!/usr/bin/env python3
"""Plot the supplied pack OCV-SOC reference and lab-derived resistance proxy."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OCV = ROOT / "models/battery_voltage_soc_map.csv"
RESISTANCE = ROOT / "models/battery_effective_resistance_map.csv"
CANDIDATE = ROOT / "models/battery_soc_resistance_candidate_cv.json"
COMPONENT = ROOT / "models/component_replay_fit.json"
OUT = ROOT / "reports/battery_soc_resistance_maps.png"


def main() -> None:
    ocv = pd.read_csv(OCV)
    resistance = pd.read_csv(RESISTANCE)
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    component = json.loads(COMPONENT.read_text(encoding="utf-8"))
    scales = [json.loads(r["parameters_json"])["scale"] for r in candidate["folds"]
              if r["model"] == "lab_R_soc_scaled" and int(r["held_out_day"]) <= 5]
    scale_mid = float(np.median(scales))
    scale_low, scale_high = float(min(scales)), float(max(scales))
    valid = resistance[resistance.resistance_usable_nonnegative.astype(bool)].copy()
    valid["candidate_scaled_r_ohm"] = valid.differential_resistance_8_to_10a_ohm * scale_mid
    valid["candidate_scaled_r_low_ohm"] = valid.differential_resistance_8_to_10a_ohm * scale_low
    valid["candidate_scaled_r_high_ohm"] = valid.differential_resistance_8_to_10a_ohm * scale_high
    r_constant = float(component["battery"]["selected_full_clean_day_r_eff_ohm"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    axes[0].plot(ocv.soc_pct, ocv.reference_voltage_25s_v, color="#245b88", marker="o", ms=4, lw=2)
    axes[0].set(xlim=(0, 100), xlabel="State of charge (%)", ylabel="Pack voltage (V)",
                title="Supplied 25-series voltage–SoC reference\nEmpirical 5% steps; not in-car true SoC or certified OCV")

    ax = axes[1]
    ax.plot(resistance.soc_pct, resistance.differential_resistance_8_to_10a_ohm,
            color="#9aa4b2", marker="o", ms=3, lw=1.2, label="2S6P lab ΔV/ΔI scaled to 25S6P")
    ax.fill_between(valid.soc_pct, valid.candidate_scaled_r_low_ohm, valid.candidate_scaled_r_high_ohm,
                    color="#1986a1", alpha=.14, label="Clean-day fold-scale range")
    ax.plot(valid.soc_pct, valid.candidate_scaled_r_ohm, color="#1986a1", marker="o", ms=4, lw=2,
            label=f"Scaled R(SOC) candidate (median scale {scale_mid:.3f})")
    ax.axhline(r_constant, color="#a64f2b", ls="--", lw=1.8,
               label=f"Current full-race constant-R model ({r_constant:.3f} Ω)")
    ax.scatter(resistance.loc[~resistance.resistance_usable_nonnegative, "soc_pct"],
               resistance.loc[~resistance.resistance_usable_nonnegative, "differential_resistance_8_to_10a_ohm"],
               color="#333333", marker="x", s=38, label="Invalid endpoint difference; excluded")
    ax.set(xlim=(0, 100), xlabel="State of charge (%)", ylabel="Effective pack resistance (Ω)",
           title="Resistance evidence and unadopted R(SOC) candidate\nDay 6 holdout worsens; no battery-temperature sweep exists")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"figure": str(OUT), "clean_fold_r_scale": scales,
                      "candidate_cv_mae_v": candidate["candidate_summaries_clean_lodo"]["lab_R_soc_scaled"]["mean_day_mae_v"],
                      "constant_r_cv_mae_v": candidate["candidate_summaries_clean_lodo"]["constant_R"]["mean_day_mae_v"],
                      "day6_candidate_mae_v": next(r["mae_v"] for r in candidate["folds"] if r["model"] == "lab_R_soc_scaled" and int(r["held_out_day"]) == 6)},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
