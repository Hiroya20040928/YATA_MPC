#!/usr/bin/env python3
"""Generate the provisioned local Grafana dashboard for YATA MPC."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "grafana/dashboards/yata_bwsc2025_shadow.json"
DS = {"type": "yesoreyeram-infinity-datasource", "uid": "yata-local-api"}
API = "http://127.0.0.1:8123/api"
GRAFANA_VERSION = "13.2.3"


def col(selector: str, label: str | None = None, kind: str = "number") -> dict[str, str]:
    return {"selector": selector, "text": label or selector, "type": kind}


def target(ref: str, path: str, columns: list[dict[str, str]], *, alias: str = "") -> dict:
    return {
        "refId": ref,
        "datasource": DS,
        "type": "json",
        "source": "url",
        "format": "table",
        "parser": "backend",
        "root_selector": "",
        "url": f"{API}/{path}",
        "url_options": {
            "method": "GET",
            "data": "",
            "body_type": "",
            "body_content_type": "",
            "body_graphql_query": "",
        },
        "columns": columns,
        "filters": [],
        "global_query_id": "",
        "alias": alias,
        "json_options": {"columnar": False, "root_is_not_array": False},
    }


def base_field(unit: str = "short", decimals: int = 1) -> dict:
    return {
        "defaults": {
            "unit": unit,
            "decimals": decimals,
            "color": {"mode": "palette-classic"},
            "mappings": [],
            "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
            "custom": {"hideFrom": {"tooltip": False, "viz": False, "legend": False}},
        },
        "overrides": [],
    }


panels: list[dict] = []


def add(panel: dict) -> None:
    panel["pluginVersion"] = GRAFANA_VERSION
    panel.setdefault("description", "")
    panels.append(panel)


def stat_panel(panel_id: int, title: str, key: str, label: str, unit: str, x: int, y: int, *, decimals: int = 1, thresholds: list[dict] | None = None) -> dict:
    field = base_field(unit, decimals)
    field["defaults"]["displayName"] = label
    if thresholds:
        field["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
        field["defaults"]["color"] = {"mode": "thresholds"}
    return {
        "id": panel_id,
        "type": "stat",
        "title": title,
        "gridPos": {"x": x, "y": y, "w": 4, "h": 4},
        "datasource": DS,
        "targets": [target("A", "current", [col(key, label)])],
        "fieldConfig": field,
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "center",
            "wideLayout": True,
            "showPercentChange": False,
            "noValue": "—",
        },
    }


def table_panel(panel_id: int, title: str, path: str, columns: list[dict[str, str]], x: int, y: int, w: int, h: int, description: str) -> dict:
    return {
        "id": panel_id,
        "type": "table",
        "title": title,
        "description": description,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [target("A", path, columns)],
        "fieldConfig": {
            "defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False}, "mappings": [], "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}, "color": {"mode": "thresholds"}},
            "overrides": [],
        },
        "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False, "reducer": ["sum"], "countRows": False, "fields": ""}},
    }


def series_target() -> dict:
    columns = [
        col("time_ms", "Time", "timestamp_epoch_ms"),
        col("speed_kmh", "speed_kmh"),
        col("target_speed_kmh", "target_speed_kmh"),
        col("battery_power_w", "battery_power_w"),
        col("pv_power_w", "pv_power_w"),
        col("battery_voltage_v", "battery_voltage_v"),
        col("battery_current_a", "battery_current_a"),
        col("energy_wh", "energy_wh"),
        col("air_temp_c", "air_temp_c"),
        col("irradiance_w_m2", "irradiance_w_m2"),
        col("wind_speed_ms", "wind_speed_ms"),
        col("measured_load_proxy_w", "measured_load_proxy_w"),
        col("predicted_vehicle_load_w", "predicted_vehicle_load_w"),
        col("predicted_pv_w", "predicted_pv_w"),
    ]
    result = target("A", "series", columns)
    result["format"] = "table"
    return result


def time_panel(panel_id: int, title: str, y: int, x: int, w: int, h: int, display: str, unit: str, series: list[tuple[str, str]], *, right_axis: tuple[str, str] | None = None, description: str = "") -> dict:
    fields = base_field(unit, 1)
    overrides = []
    for name, label in series:
        overrides.append({
            "matcher": {"id": "byName", "options": name},
            "properties": [{"id": "displayName", "value": label}],
        })
    if right_axis:
        name, axis_unit = right_axis
        overrides.append({
            "matcher": {"id": "byName", "options": name},
            "properties": [
                {"id": "displayName", "value": "風速"},
                {"id": "unit", "value": axis_unit},
                {"id": "custom.axisPlacement", "value": "right"},
            ],
        })
        fields["defaults"]["custom"] = {
            "drawStyle": "line", "lineInterpolation": "linear", "lineWidth": 2,
            "fillOpacity": 8, "showPoints": "never", "spanNulls": False,
            "axisPlacement": "left", "axisLabel": display,
            "scaleDistribution": {"type": "linear"},
            "hideFrom": {"tooltip": False, "viz": False, "legend": False},
        }
    else:
        fields["defaults"]["custom"] = {
            "drawStyle": "line", "lineInterpolation": "linear", "lineWidth": 2,
            "fillOpacity": 8, "showPoints": "never", "spanNulls": False,
            "axisPlacement": "auto", "axisLabel": display,
            "scaleDistribution": {"type": "linear"},
            "hideFrom": {"tooltip": False, "viz": False, "legend": False},
        }
    fields["overrides"] = overrides
    return {
        "id": panel_id,
        "type": "timeseries",
        "title": title,
        "description": description,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [series_target()],
        "fieldConfig": fields,
        "options": {
            "tooltip": {"mode": "multi", "sort": "desc"},
            "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom", "calcs": ["lastNotNull"]},
        },
    }


add({
    "id": 1,
    "type": "text",
    "title": "運用ステータス",
    "gridPos": {"x": 0, "y": 0, "w": 24, "h": 3},
    "options": {
        "mode": "markdown",
        "content": "## BWSC2025 ログ再生 / MPC シャドー提案　|　再生 1×　|　**実測テレメトリとMPC提案を分離表示**\n\n本画面は2025年の43,963行をローカル再生しています。MPC値は実車に出力した指令ではなく、Grafana上からも車両制御は行いません。地図は公式ルートの再構成線上に再生中のログ距離を投影します。",
    },
})

kpis = [
    ("速度（ログ実測）", "speed_kmh", "実測速度", "velocityKmh", 1, None),
    ("上位MPC目標（シャドー）", "target_speed_kmh", "巡航目標", "velocityKmh", 1, None),
    ("推定バッテリー残量", "soc_estimate_pct", "残量", "percent", 1, [{"color": "red", "value": None}, {"color": "orange", "value": 15}, {"color": "green", "value": 30}]),
    ("推定蓄電エネルギー", "energy_wh", "蓄電量", "wattHour", 0, None),
    ("バッテリー電力（符号付ログ値）", "battery_power_w", "Battery_Watt", "watt", 0, None),
    ("PV合算出力（ZPログ）", "pv_power_w", "PV出力", "watt", 0, None),
    ("HVバス電圧", "battery_voltage_v", "電圧", "volt", 1, None),
    ("バッテリー電流", "battery_current_a", "電流", "amp", 2, None),
    ("外気温（Open-Meteo再解析）", "air_temp_c", "気温", "celsius", 1, None),
    ("風速（Open-Meteo再解析）", "wind_speed_ms", "風速", "velocityms", 1, None),
    ("瞬時日射（Open-Meteo再解析）", "irradiance_w_m2", "日射", "wattpermsquared", 0, None),
    ("予報ラン経過時間", "forecast_age_h", "予報経過", "hour", 1, [{"color": "green", "value": None}, {"color": "yellow", "value": 10}, {"color": "red", "value": 12}]),
]
panel_id = 2
for row, y in enumerate((3, 7)):
    for col_ix in range(6):
        title, key, label, unit, decimals, thresholds = kpis[row * 6 + col_ix]
        add(stat_panel(panel_id, title, key, label, unit, col_ix * 4, y, decimals=decimals, thresholds=thresholds))
        panel_id += 1

map_panel = {
    "id": panel_id,
    "type": "geomap",
    "title": "BWSCルート / 現在位置 / 記録イベント",
    "description": "Geomapの車両点は5秒ごとに更新します。ルートと大会記録イベントは公式ルート資料に基づく固定レイヤーです。地図背景はOpenStreetMapタイルを使用します。",
    "gridPos": {"x": 0, "y": 11, "w": 15, "h": 18},
    "datasource": DS,
    "targets": [target("A", "current", [
        col("latitude", "latitude"), col("longitude", "longitude"),
        col("speed_kmh", "車速 km/h"), col("distance_km", "距離 km"),
        col("replay_day", "Day"), col("race_time_local", "大会ログ時刻", "string"),
        col("target_speed_kmh", "MPC目標 km/h"), col("soc_estimate_pct", "推定残量 %"),
    ])],
    "fieldConfig": {"defaults": {"custom": {"hideFrom": {"tooltip": False, "viz": False, "legend": False}}, "mappings": [], "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}, "color": {"mode": "palette-classic"}}, "overrides": []},
    "options": {
        "view": {"id": "coords", "lat": -25.7, "lon": 134.0, "zoom": 4, "allLayers": True},
        "controls": {"showZoom": True, "mouseWheelZoom": True, "showAttribution": True, "showScale": True, "showMeasure": False, "showDebug": False},
        "tooltip": {"mode": "details"},
        "basemap": {"config": {}, "name": "OpenStreetMap", "type": "osm-standard"},
        "layers": [
            {"type": "geojson", "name": "走行ルートとイベント地点", "config": {"src": f"{API}/route.geojson", "rules": []}, "tooltip": True},
            {
                "type": "markers", "name": "再生中の車両位置", "filterData": {"id": "byRefId", "options": "A"},
                "location": {"mode": "coords", "latitude": "latitude", "longitude": "longitude"},
                "config": {"showLegend": True, "style": {"size": {"fixed": 10, "min": 8, "max": 14}, "color": {"fixed": "#00D4FF"}, "opacity": 1, "symbol": {"mode": "fixed", "fixed": "img/icons/marker/circle.svg"}, "rotation": {"fixed": 0, "mode": "mod", "min": -360, "max": 360}}},
                "tooltip": True,
            },
        ],
    },
}
add(map_panel)
panel_id += 1

add(table_panel(panel_id, "制御器の状態 / 判断根拠", "decision", [col("項目", "項目", "string"), col("値", "値", "string"), col("補足", "補足", "string")], 15, 11, 9, 10, "上位30分MPC、下位30秒MPC、予報run時刻、予備制約を表示。")); panel_id += 1
add(table_panel(panel_id, "運用確認 / 現在ログの警報判定", "alerts", [col("severity", "判定", "string"), col("status", "状態", "string"), col("item", "内容", "string"), col("distance_km", "距離 km"), col("solver", "上位ソルバー", "string")], 15, 21, 9, 8, "実測ログのイベント、MPC求解状態、暫定15%残量条件を確認します。警報はログとモデル出力から算出し、車両へ指令しません。")); panel_id += 1

add(time_panel(panel_id, "走行速度と上位MPCの巡航目標", 29, 0, 12, 9, "速度 [km/h]", "velocityKmh", [("speed_kmh", "ログ実測速度"), ("target_speed_kmh", "MPC目標（シャドー）")], description="直近1時間のログ再生。MPC目標は上位再計画間では直前値を保持します。")); panel_id += 1
add(time_panel(panel_id, "HVバッテリー電力とPV出力", 29, 12, 12, 9, "電力 [W]", "watt", [("battery_power_w", "Battery_Watt（符号付ログ値）"), ("pv_power_w", "Solar_Watt / ZPログ")], description="両系列は既存ログの値をそのまま表示します。Battery_Wattの正負は計測器の符号定義を別途確認してください。")); panel_id += 1
add(time_panel(panel_id, "電力モデル実測整合（負荷・PV）", 38, 0, 12, 9, "電力 [W]", "watt", [("measured_load_proxy_w", "負荷代理値（実測算出）"), ("predicted_vehicle_load_w", "車両モデル推定"), ("pv_power_w", "PV実測ログ"), ("predicted_pv_w", "PVモデル推定")], description="測定値とモデル予測値を重ね、走行中の差を視認します。負荷代理値はHVバスログ由来で、独立したトルク計測値ではありません。")); panel_id += 1
add(time_panel(panel_id, "推定バッテリー蓄電量", 38, 12, 12, 9, "蓄電量 [Wh]", "wattHour", [("energy_wh", "3011 Wh設計容量基準の推定蓄電量")], description="3011 Whは理論容量。SOCは電圧アンカーとログ電力積分による推定で、BMS真値ではありません。")); panel_id += 1

weather_series = series_target()
add({
    "id": panel_id,
    "type": "timeseries",
    "title": "気温と風速（Open-Meteo履歴推定）",
    "description": "モジュール温度センサはありません。ここに表示する気温と風速はログ時刻・距離に対応するOpen-Meteo履歴値です。",
    "gridPos": {"x": 0, "y": 47, "w": 12, "h": 9},
    "datasource": DS,
    "targets": [weather_series],
    "fieldConfig": {
        "defaults": {"unit": "celsius", "decimals": 1, "color": {"mode": "palette-classic"}, "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never", "axisPlacement": "left", "axisLabel": "気温 [°C]", "scaleDistribution": {"type": "linear"}, "hideFrom": {"tooltip": False, "viz": False, "legend": False}}, "mappings": [], "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}},
        "overrides": [
            {"matcher": {"id": "byName", "options": "air_temp_c"}, "properties": [{"id": "displayName", "value": "外気温"}]},
            {"matcher": {"id": "byName", "options": "wind_speed_ms"}, "properties": [{"id": "displayName", "value": "風速"}, {"id": "unit", "value": "velocityms"}, {"id": "custom.axisPlacement", "value": "right"}]},
        ],
    },
    "options": {"tooltip": {"mode": "multi", "sort": "desc"}, "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom", "calcs": ["lastNotNull"]}},
}); panel_id += 1

add(time_panel(panel_id, "瞬時日射量（Open-Meteo履歴推定）", 47, 12, 12, 9, "日射 [W/m²]", "wattpermsquared", [("irradiance_w_m2", "短波放射・瞬時値")], description="PV実測電力と照合する外部気象の日射指標です。車体面POA日射を直接計測した値ではありません。")); panel_id += 1

add(table_panel(panel_id, "BWSC2025 ルートイベント / トラブル・停止記録", "events", [
    col("day", "Day", "string"), col("start_local", "時刻", "string"), col("event_type", "種別", "string"),
    col("event_label", "記録", "string"), col("distance_m", "距離 m"), col("distance_quality", "位置根拠", "string"), col("source", "出典", "string"),
], 0, 56, 24, 12, "公式ルートノート・大会タイムラインから整理した2025年記録です。イベント位置は人間運転ログ時の記録で、MPC走行後の発生時刻/距離を予測したものではありません。")); panel_id += 1

dashboard = {
    "id": None,
    "uid": "yata-bwsc2025-local",
    "title": "YATA MPC | BWSC2025 Shadow Operations",
    "description": "BWSC2025 historical replay and two-layer MPC shadow advisory. No vehicle actuation is connected.",
    "tags": ["YATA", "BWSC2025", "local", "replay", "shadow-MPC"],
    "timezone": "browser",
    "schemaVersion": 41,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-1h", "to": "now"},
    "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"], "time_options": ["15m", "1h", "3h", "6h"]},
    "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"}]},
    "panels": panels,
    "links": [],
    "liveNow": False,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} ({len(panels)} panels)")
