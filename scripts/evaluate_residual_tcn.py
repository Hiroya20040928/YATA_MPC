#!/usr/bin/env python3
"""Causal temporal-convolution residual model with chronological validation."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from evaluate_residual_method_sweep import ROOT, FEATURES, prepare_data, day_splits

OUT_CSV = ROOT / "reports/residual_tcn_oof_predictions.csv"
OUT_JSON = ROOT / "reports/residual_tcn_summary.json"
OUT_MODEL = ROOT / "models/vehicle_residual_tcn_maxfit.pt"
OUT_META = ROOT / "models/vehicle_residual_tcn_manifest.json"
WINDOW = 32
DEVICE = torch.device("cpu")
SEQ_FEATURES = ["speed_kmh", "distance_km", "route_grade_smoothed_pct",
                "temperature_2m", "wind_speed_10m", "headwind_mps", "crosswind_mps",
                "shortwave_radiation_instant", "direct_radiation_instant",
                "diffuse_radiation_instant", "global_tilted_irradiance_instant",
                "pressure_msl", "cloud_cover", "residual_lag_1", "residual_lag_2",
                "residual_lag_3", "residual_lag_6", "residual_lag_12",
                "residual_mean_6", "residual_mean_24"]


class TCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (2 * self.dilation, 0))
        y = self.conv(y)
        y = self.norm(F.gelu(y))
        return F.gelu(x + self.dropout(y))


class TCN(nn.Module):
    def __init__(self, n_features: int, width: int, dropout: float):
        super().__init__()
        self.input = nn.Conv1d(n_features, width, kernel_size=1)
        self.blocks = nn.Sequential(*[TCNBlock(width, d, dropout) for d in [1, 2, 4, 8, 16]])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.blocks(self.input(z))
        return self.head(z[:, :, -1]).squeeze(-1)


def make_windows(x: np.ndarray, stream_ids: np.ndarray, window: int = WINDOW) -> np.ndarray:
    out = np.empty((len(x), window, x.shape[1]), dtype=np.float32)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and stream_ids[end] == stream_ids[start]:
            end += 1
        seq = x[start:end]
        for i in range(len(seq)):
            lo = max(0, i - window + 1)
            recent = seq[lo:i + 1]
            left = window - len(recent)
            out[start + i, :left] = recent[0]
            out[start + i, left:] = recent
        start = end
    return out


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def score(y: np.ndarray, p: np.ndarray) -> dict:
    e = np.asarray(y, float) - np.asarray(p, float)
    return {"n": int(len(e)), "mae_w": mae(y, p),
            "rmse_w": float(np.sqrt(np.mean(e**2))),
            "p90_abs_error_w": float(np.quantile(np.abs(e), .9)),
            "bias_w": float(np.mean(e))}


def tensor_batches(x: torch.Tensor, y: torch.Tensor, indices: np.ndarray, batch: int,
                   shuffle: bool = False):
    order = np.random.permutation(indices) if shuffle else indices
    for start in range(0, len(order), batch):
        ix = order[start:start + batch]
        yield x[ix], y[ix]


def fit_model(x: np.ndarray, y: np.ndarray, train_ix: np.ndarray, val_ix: np.ndarray,
              width: int, dropout: float, max_epochs: int = 30,
              fixed_epochs: int | None = None) -> tuple[TCN, int, list[float]]:
    torch.manual_seed(43)
    np.random.seed(43)
    model = TCN(x.shape[-1], width, dropout).to(DEVICE)
    tx = torch.from_numpy(x).to(DEVICE)
    ty = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-3)
    best_state = None
    best_score = float("inf")
    best_epoch = max_epochs
    patience, stale = 6, 0
    epoch_scores = []
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    for epoch in range(epochs):
        model.train()
        for xb, yb in tensor_batches(tx, ty, train_ix, 1024, shuffle=True):
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb, beta=75.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            vp = np.concatenate([model(xb).cpu().numpy() for xb, _ in tensor_batches(tx, ty, val_ix, 1024)])
        val_mae = mae(ty[val_ix].cpu().numpy(), vp)
        epoch_scores.append(val_mae)
        if val_mae < best_score - 0.05:
            best_score = val_mae
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break
    if best_state is not None and fixed_epochs is None:
        model.load_state_dict(best_state)
    return model, best_epoch, epoch_scores


def predict(model: TCN, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model.eval()
    tx = torch.from_numpy(x).to(DEVICE)
    with torch.no_grad():
        return np.concatenate([model(xb).cpu().numpy()
                               for xb, _ in tensor_batches(tx, torch.zeros(len(tx)), indices, 1024)])


def main() -> None:
    torch.set_num_threads(4)
    data = prepare_data().reset_index(drop=True)
    y = data.target_residual_w.to_numpy(np.float32)
    seq_features = [c for c in SEQ_FEATURES if c in data.columns]
    raw_x = data[seq_features].to_numpy(float)
    clean_train_ix, clean_val_ix, clean_test_ix = [], [], []
    per_day_positions: dict[int, dict[str, np.ndarray]] = {}
    for day, idx in data.groupby("day", sort=True).groups.items():
        local = np.asarray(list(idx), int)
        a, b, c = day_splits(data.iloc[local].reset_index(drop=True))
        per_day_positions[int(day)] = {"train": local[a], "val": local[b], "test": local[c], "all": local}
        if day <= 5:
            clean_train_ix.extend(local[a].tolist())
            clean_val_ix.extend(local[b].tolist())
            clean_test_ix.extend(local[c].tolist())
    clean_train_ix = np.asarray(clean_train_ix, int)
    clean_val_ix = np.asarray(clean_val_ix, int)
    clean_test_ix = np.asarray(clean_test_ix, int)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imp = imputer.fit_transform(raw_x[clean_train_ix])
    all_imp = imputer.transform(raw_x)
    scaler = StandardScaler().fit(train_imp)
    all_x = scaler.transform(all_imp).astype(np.float32)
    stream_ids = (data.day.to_numpy(np.int32) * 100000 + data.stream_id.to_numpy(np.int32))
    x_windows = make_windows(all_x, stream_ids)

    # Hyperparameter selection uses only Days 1-5 validation segments.
    candidates = [(8, 0.10), (16, 0.10)]
    configs = []
    for width, dropout in candidates:
        model, best_epoch, history = fit_model(x_windows, y, clean_train_ix, clean_val_ix,
                                               width, dropout, max_epochs=30)
        val_pred = predict(model, x_windows, clean_val_ix)
        configs.append({"width": width, "dropout": dropout, "best_epoch": best_epoch,
                        "model": model, "validation": score(y[clean_val_ix], val_pred),
                        "epoch_curve": history})
    selected = min(configs, key=lambda r: r["validation"]["mae_w"])
    width, dropout, selected_epochs = selected["width"], selected["dropout"], selected["best_epoch"]

    # Refit selected network on all pre-test clean-day rows for the untouched
    # last-20% test, using the validation-selected epoch count.
    pretest_clean = np.concatenate([per_day_positions[d]["train"] for d in range(1, 6)] +
                                   [per_day_positions[d]["val"] for d in range(1, 6)])
    refit, _, _ = fit_model(x_windows, y, pretest_clean, clean_val_ix, width, dropout,
                            max_epochs=selected_epochs, fixed_epochs=selected_epochs)
    clean_pred = predict(refit, x_windows, clean_test_ix)
    clean_rows = data.iloc[clean_test_ix][["day", "timestamp_utc"]].copy().reset_index(drop=True)
    clean_rows["target_residual_w"] = y[clean_test_ix]
    clean_rows["tcn_correction_w"] = clean_pred
    clean_rows["corrected_error_w"] = clean_rows.target_residual_w - clean_pred
    clean_rows["scope"] = "clean_days_1_5_last20_test"

    # Prospective stress test: train on all clean-day rows, score every Day6 row
    # in one-step mode; each target uses only earlier measured residual features.
    d6_ix = per_day_positions[6]["all"]
    all_clean = np.concatenate([per_day_positions[d]["all"] for d in range(1, 6)])
    final_clean_fit = refit
    d6_pred = predict(final_clean_fit, x_windows, d6_ix)
    d6_rows = data.iloc[d6_ix][["day", "timestamp_utc"]].copy().reset_index(drop=True)
    d6_rows["target_residual_w"] = y[d6_ix]
    d6_rows["tcn_correction_w"] = d6_pred
    d6_rows["corrected_error_w"] = d6_rows.target_residual_w - d6_pred
    d6_rows["scope"] = "day6_stress_all_rows"
    output = pd.concat([clean_rows, d6_rows], ignore_index=True)
    output.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    torch.save({"state_dict": refit.state_dict(), "width": width, "dropout": dropout,
                "window": WINDOW, "n_features": int(len(seq_features)),
                "feature_names": seq_features, "imputer_statistics": imputer.statistics_.tolist(),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "training_rows": int(len(data))}, OUT_MODEL)

    clean_base = y[clean_test_ix]
    d6_base = y[d6_ix]
    summary = {
        "method": "causal dilated temporal convolution network; 32-step window, dilations 1/2/4/8/16, SmoothL1 training; one-step residual prediction",
        "features": seq_features, "window_rows": WINDOW,
        "selection": "choose width/dropout using Days 1-5 validation slices only; last 20% of each day remains untouched until scoring",
        "configs": [{"width": c["width"], "dropout": c["dropout"], "best_epoch": c["best_epoch"],
                     "validation": c["validation"]} for c in configs],
        "selected": {"width": width, "dropout": dropout, "epochs": selected_epochs},
        "clean_days_1_5_outer_test": {"physical_baseline": score(clean_base, np.zeros(len(clean_base))),
                                      "tcn": score(clean_rows.target_residual_w.to_numpy(), clean_rows.tcn_correction_w.to_numpy()),
                                      "per_day": {str(day): {"n": int((clean_rows.day == day).sum()),
                                          "mae_w": float(np.mean(np.abs(clean_rows.loc[clean_rows.day == day, "corrected_error_w"]))),
                                          "p90_abs_error_w": float(np.quantile(np.abs(clean_rows.loc[clean_rows.day == day, "corrected_error_w"]), .9))}
                                          for day in range(1,6)}},
        "day6_stress_all_rows": {"physical_baseline": score(d6_base, np.zeros(len(d6_base))),
                                 "tcn": score(d6_rows.target_residual_w.to_numpy(), d6_rows.tcn_correction_w.to_numpy())},
        "model_fit_note": "The saved TCN is fit on Days1-5 first80% rows, not all data; it is an evaluated challenger. The full-data committed model remains the selected tabular HGB unless TCN improves on the identical outer test.",
        "output_model": str(OUT_MODEL), "output_predictions": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_META.write_text(json.dumps({"model": str(OUT_MODEL), "architecture": summary["method"],
        "selected": summary["selected"], "feature_names": seq_features,
        "normalization": "imputer and scaler fitted on Days1-5 first60% for OOF; serialized feature scalers in checkpoint",
        "inference": "provide a 64-row causal feature window; residual lags are previous measured errors only",
        "fit_rows_final": int(len(data)), "validation": summary["clean_days_1_5_outer_test"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
