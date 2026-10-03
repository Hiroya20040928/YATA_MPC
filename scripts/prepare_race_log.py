#!/usr/bin/env python3
"""Merge BWSC2025 ZP exports and assign anchor-calibrated route positions."""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TZ = ZoneInfo("Australia/Darwin")
RACE_DATES = {d: date(2025, 8, 23 + d) for d in range(1, 7)}
SENSOR_COLS = ["Solar[A]", "Battery[A]", "Main_Volt[V]", "Solar_Watt[W]", "Battery_Watt[W]", "Sub_Volt[V]", "Speed[kmh]"]


def read_raw(path: Path) -> tuple[pd.DataFrame, int | None]:
    raw_bytes = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp932", "utf-8", "latin1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        raise ValueError(f"Cannot decode {path}")
    lines = text.splitlines()
    expected = None
    for line in lines[:10]:
        m = re.search(r"(?:データ数|データ件数|data\s*count)\s*[=：:]\s*,?\s*(\d+)", line, re.I)
        if m:
            expected = int(m.group(1))
            break
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("No,")), None)
    if header_idx is None:
        raise ValueError(f"CSV header starting No, not found in {path}")
    # pandas preserves the export's wide/sparse ancillary columns and tolerates missing trailing fields.
    frame = pd.read_csv(path, encoding="cp932", skiprows=header_idx, dtype=str, keep_default_na=False)
    names = []
    seen: dict[str, int] = {}
    for idx, col in enumerate(frame.columns):
        col = str(col).strip()
        if not col:
            col = f"source_col_{idx}"
        seen[col] = seen.get(col, 0) + 1
        names.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    frame.columns = names
    # ZP's export can append independent battery summaries after the main log table.
    # Keep the primary contiguous 1..N log block; the metadata row count is authoritative.
    numeric_no = pd.to_numeric(frame["No"], errors="coerce")
    frame = frame.loc[numeric_no.notna()].copy()
    if expected is not None:
        frame = frame.iloc[:expected]
    else:
        ids = pd.to_numeric(frame["No"], errors="coerce").to_numpy()
        cut = next((i for i in range(1, len(ids)) if ids[i] != ids[i - 1] + 1), len(ids))
        frame = frame.iloc[:cut]
    frame = frame.reset_index(drop=True)
    return frame, expected


def parse_time(value: str) -> time:
    parts = str(value).strip().split(":")
    return time(int(parts[0]), int(parts[1]), int(float(parts[2])))


def make_datetime(day: int, value: str) -> datetime:
    return datetime.combine(RACE_DATES[day], parse_time(value), tzinfo=LOCAL_TZ)


def load_anchors() -> pd.DataFrame:
    a = pd.read_csv(ROOT / "data/input/route/anchors.csv", keep_default_na=False)
    for col in ("day", "distance_m", "use_for_distance", "uncertainty_m"):
        a[col] = pd.to_numeric(a[col].replace("", np.nan), errors="coerce")
    a["anchor_dt"] = [make_datetime(int(r.day), r.anchor_time_local) for r in a.itertuples()]
    return a


def anchor_rows_for_day(frame: pd.DataFrame, anchors: pd.DataFrame, day: int) -> list[dict]:
    rows = []
    subset = anchors[(anchors.day == day) & (anchors.use_for_distance == 1)].sort_values("anchor_dt")
    dts = pd.to_datetime(frame["timestamp_local"], utc=True).dt.tz_convert(LOCAL_TZ)
    for a in subset.itertuples():
        delta = (dts - a.anchor_dt).abs().dt.total_seconds()
        idx = int(delta.idxmin())
        sec = float(delta.iloc[idx])
        if sec > 30:
            raise ValueError(f"Anchor {a.event_id} has no telemetry within 30 sec (nearest {sec}s)")
        rows.append({"idx": idx, "anchor_time": a.anchor_time_local, "event_id": a.event_id,
                     "distance_m": float(a.distance_m), "uncertainty_m": None if pd.isna(a.uncertainty_m) else float(a.uncertainty_m),
                     "confidence": a.confidence, "match_offset_s": sec})
    # Several event boundaries can fall in one 5 s sample; keep a deterministic single row only when distances agree.
    collapsed = []
    for r in rows:
        if collapsed and collapsed[-1]["idx"] == r["idx"]:
            if abs(collapsed[-1]["distance_m"] - r["distance_m"]) > 1:
                raise ValueError(f"Conflicting anchors map to same telemetry row: {collapsed[-1]} and {r}")
            collapsed[-1]["event_id"] += ";" + r["event_id"]
        else:
            collapsed.append(r)
    if len(collapsed) < 2:
        raise ValueError(f"Need at least two anchors for day {day}")
    return collapsed


def assign_distance(frame: pd.DataFrame, anchors: pd.DataFrame, day: int) -> tuple[pd.DataFrame, list[dict]]:
    out = frame.copy()
    speed = pd.to_numeric(out["Speed[kmh]"], errors="coerce").fillna(0).clip(lower=0).to_numpy() / 3.6
    timestamp_utc = pd.to_datetime(out["timestamp_local"], utc=True)
    dt = timestamp_utc.diff().dt.total_seconds().to_numpy(dtype=float, copy=True)
    dt[0] = np.median(dt[1:][dt[1:] > 0]) if np.any(dt[1:] > 0) else 5.0
    dt = np.clip(dt, 0, 60)
    # Trapezoidal speed integration, later calibrated to each documented anchor pair.
    raw_step = (np.r_[speed[0], (speed[1:] + speed[:-1]) / 2] * dt)
    raw_cum = np.cumsum(raw_step)
    ar = anchor_rows_for_day(out, anchors, day)
    dist = np.full(len(out), np.nan)
    seg_id = np.full(len(out), "", dtype=object)
    scale = np.full(len(out), np.nan)
    confidence = np.full(len(out), "", dtype=object)
    uncertainty = np.full(len(out), np.nan)
    for si, (left, right) in enumerate(zip(ar[:-1], ar[1:]), start=1):
        i0, i1 = left["idx"], right["idx"]
        if i1 <= i0:
            raise ValueError(f"Non-increasing anchor rows day {day}: {left} -> {right}")
        target = right["distance_m"] - left["distance_m"]
        if target < 0:
            raise ValueError(f"Distance reverses between anchors day {day}: {left} -> {right}")
        local = raw_cum[i0:i1 + 1] - raw_cum[i0]
        raw_delta = float(local[-1])
        if target == 0:
            profile = np.zeros(len(local))
            factor = 0.0
        elif raw_delta > 0.5:
            factor = target / raw_delta
            profile = local * factor
        else:
            # If ZP speed is absent/zero in an interval with known travel, time-linear interpolation is the only supported fallback.
            factor = math.nan
            profile = np.linspace(0, target, len(local))
        dist[i0:i1 + 1] = left["distance_m"] + profile
        seg_id[i0:i1 + 1] = f"D{day}_S{si:02d}:{left['event_id']}->{right['event_id']}"
        scale[i0:i1 + 1] = factor
        confidence[i0:i1 + 1] = "high" if left["confidence"] == right["confidence"] == "high" else "medium_or_lower"
        uncertainty[i0:i1 + 1] = max([x for x in (left["uncertainty_m"], right["uncertainty_m"]) if x is not None] or [np.nan])
    # Keep edge samples at the nearest endpoint and mark them as extrapolated.
    first, last = ar[0], ar[-1]
    dist[:first["idx"]] = first["distance_m"]
    dist[last["idx"] + 1:] = last["distance_m"]
    seg_id[:first["idx"]] = f"D{day}_EDGE_BEFORE:{first['event_id']}"
    seg_id[last["idx"] + 1:] = f"D{day}_EDGE_AFTER:{last['event_id']}"
    confidence[:first["idx"]] = confidence[last["idx"] + 1:] = "endpoint_hold"
    uncertainty[:first["idx"]] = first["uncertainty_m"] if first["uncertainty_m"] is not None else np.nan
    uncertainty[last["idx"] + 1:] = last["uncertainty_m"] if last["uncertainty_m"] is not None else np.nan
    dist[first["idx"]] = first["distance_m"]
    dist[last["idx"]] = last["distance_m"]
    out["distance_from_darwin_m"] = dist
    out["distance_fit_segment"] = seg_id
    out["distance_speed_scale_factor"] = scale
    out["distance_anchor_confidence"] = confidence
    out["distance_anchor_uncertainty_m"] = uncertainty
    out["distance_is_endpoint_hold"] = ((np.arange(len(out)) < first["idx"]) | (np.arange(len(out)) > last["idx"])).astype(int)
    if np.nanmin(np.diff(dist)) < -1e-6:
        raise ValueError(f"Distance is not monotonic on day {day}")
    fit_rows = []
    for si, (left, right) in enumerate(zip(ar[:-1], ar[1:]), start=1):
        fit_rows.append({"day": day, "segment": si, "from_anchor": left["event_id"], "to_anchor": right["event_id"],
                         "from_distance_m": left["distance_m"], "to_distance_m": right["distance_m"],
                         "observed_route_delta_m": right["distance_m"] - left["distance_m"],
                         "raw_speed_integral_m": float(raw_cum[right["idx"]] - raw_cum[left["idx"]]),
                         "multiplicative_scale": scale[left["idx"]], "from_match_offset_s": left["match_offset_s"],
                         "to_match_offset_s": right["match_offset_s"], "confidence": "high" if left["confidence"] == right["confidence"] == "high" else "medium_or_lower"})
    return out, fit_rows


def main() -> None:
    clean_dir = ROOT / "data/input/clean_logs"
    raw_dir = ROOT / "data/input/raw_logs"
    anchors = load_anchors()
    merged_days, geo_days, fit = [], [], []
    quality = {"method": "5-second speed trapezoidal integration calibrated piecewise to route/time anchors; zero-distance anchor spans held stationary", "timezone": "Australia/Darwin (ACST, UTC+09:30)", "race_dates": {str(k): str(v) for k, v in RACE_DATES.items()}, "days": []}
    for day in range(1, 7):
        raw, expected = read_raw(raw_dir / f"day{day}.csv")
        clean = pd.read_csv(clean_dir / f"day{day}.csv")
        if len(raw) != len(clean):
            raise ValueError(f"Raw vs clean row count mismatch day {day}: {len(raw)} vs {len(clean)}")
        if expected is not None and expected != len(raw):
            raise ValueError(f"Export metadata row count mismatch day {day}: {expected} vs {len(raw)}")
        raw_sensor = raw.reindex(columns=SENSOR_COLS).apply(pd.to_numeric, errors="coerce")
        clean_sensor = clean.reindex(columns=SENSOR_COLS).apply(pd.to_numeric, errors="coerce")
        max_diffs = {c: float((raw_sensor[c] - clean_sensor[c]).abs().max()) for c in SENSOR_COLS}
        if any(v > 1e-8 for v in max_diffs.values()):
            raise ValueError(f"Raw-clean sensor values differ day {day}: {max_diffs}")
        day_frame = pd.concat([raw.reset_index(drop=True), clean[["time"]].rename(columns={"time": "time_local"})], axis=1)
        day_frame.insert(0, "day", day)
        day_frame.insert(1, "race_date", str(RACE_DATES[day]))
        day_frame.insert(2, "source_file", f"day{day}.csv")
        day_frame.insert(3, "source_row", np.arange(1, len(day_frame) + 1))
        day_frame["timestamp_local"] = [make_datetime(day, v).isoformat() for v in day_frame["time_local"]]
        day_frame["timestamp_utc"] = pd.to_datetime(day_frame["timestamp_local"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        day_frame["sample_interval_s"] = pd.to_datetime(day_frame["timestamp_local"], utc=True).diff().dt.total_seconds().fillna(0)
        for col in SENSOR_COLS:
            day_frame[col] = pd.to_numeric(day_frame[col], errors="coerce")
        day_frame["pv_power_w"] = day_frame["Solar_Watt[W]"]
        day_frame["battery_power_signed_w"] = day_frame["Battery_Watt[W]"]
        day_frame["net_electrical_power_proxy_w"] = day_frame["Solar_Watt[W]"] + day_frame["Battery_Watt[W]"]
        # Event context is separately represented for auditability; overlapping event types are semicolon joined.
        events = pd.read_csv(ROOT / "data/input/route/events.csv", keep_default_na=False)
        event_rows = events[events.day == day]
        labels = [[] for _ in range(len(day_frame))]
        local_times = pd.to_datetime(day_frame["timestamp_local"], utc=True).dt.tz_convert(LOCAL_TZ)
        second_of_day = (local_times.dt.hour * 3600 + local_times.dt.minute * 60 + local_times.dt.second).to_numpy()
        for ev in event_rows.itertuples():
            def sod(hms: str) -> int:
                parsed = parse_time(hms)
                return parsed.hour * 3600 + parsed.minute * 60 + parsed.second
            begin = sod(ev.start_local)
            finish = sod(ev.end_local) if ev.end_local else begin
            mask = (second_of_day >= begin) & (second_of_day <= finish)
            flag = ev.event_type + ":" + ev.event_id
            for idx in np.flatnonzero(mask):
                labels[idx].append(flag)
        labels = [";".join(value) for value in labels]
        day_frame["event_flags"] = labels
        day_frame["event_phase"] = [x.split(":", 1)[0] if x else "road_or_logging" for x in labels]
        merged_days.append(day_frame)
        geo, rows = assign_distance(day_frame, anchors, day)
        fit.extend(rows)
        geo_days.append(geo)
        gaps = day_frame.loc[day_frame.sample_interval_s > 5.01, ["time_local", "sample_interval_s"]]
        quality["days"].append({"day": day, "date": str(RACE_DATES[day]), "rows_raw": len(raw), "rows_clean": len(clean),
                                "metadata_expected_rows": expected, "source_clean_sensor_max_abs_diffs": max_diffs,
                                "median_sample_interval_s": float(day_frame.sample_interval_s.iloc[1:].median()),
                                "max_sample_interval_s": float(day_frame.sample_interval_s.max()),
                                "time_gaps_over_5s": [{"gap_ends_at_local": r.time_local, "gap_seconds": float(r.sample_interval_s)} for r in gaps.itertuples()],
                                "zero_speed_rows": int((day_frame["Speed[kmh]"] == 0).sum()),
                                "distance_m_min": float(geo.distance_from_darwin_m.min()), "distance_m_max": float(geo.distance_from_darwin_m.max())})
    merged = pd.concat(merged_days, ignore_index=True, sort=False)
    geo = pd.concat(geo_days, ignore_index=True, sort=False)
    route = pd.read_csv(ROOT / "data/input/route/route_100m_dem.csv")
    if geo.distance_from_darwin_m.min() < route.distance_m.min() or geo.distance_from_darwin_m.max() > route.distance_m.max():
        raise ValueError("Anchor-calibrated distance exceeds supplied route track")
    geo["route_latitude"] = np.interp(geo.distance_from_darwin_m, route.distance_m, route.lat)
    geo["route_longitude"] = np.interp(geo.distance_from_darwin_m, route.distance_m, route.lon)
    geo["route_elevation_dem_m"] = np.interp(geo.distance_from_darwin_m, route.distance_m, route.elev_dem_m)
    geo["route_grade_smoothed_pct"] = np.interp(geo.distance_from_darwin_m, route.distance_m, route.grade_smoothed_pct.fillna(0))
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    merged.to_csv(ROOT / "data/processed/race_days_merged.csv", index=False, encoding="utf-8-sig")
    geo.to_csv(ROOT / "data/processed/race_days_distance_geo.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fit).to_csv(ROOT / "reports/distance_anchor_fit.csv", index=False, encoding="utf-8-sig")
    quality["total_rows"] = len(merged)
    quality["unique_fields"] = list(merged.columns)
    quality["distance_assignment_note"] = "Distances between anchors follow the observed ZP speed profile scaled to the anchor delta. This preserves timing/relative motion but is not independent GPS truth; stop times are pinned only when supported by paired equal-distance anchors. Anchor mismatch uncertainty is retained per row."
    (ROOT / "reports/data_quality_summary.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": len(merged), "merged_csv": str(ROOT / "data/processed/race_days_merged.csv"),
                      "distance_geo_csv": str(ROOT / "data/processed/race_days_distance_geo.csv"), "anchor_segments": len(fit)}, indent=2))


if __name__ == "__main__":
    main()
