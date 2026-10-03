#!/usr/bin/env python3
"""Fetch the full Open-Meteo Weather Forecast API dataset for one coordinate."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "YATA-MPC-OpenMeteo/0.1 (BWSC2027 weather data collector)"
LOG = logging.getLogger("yata_mpc.openmeteo")

# Weather Forecast API hourly variables, including all documented pressure
# levels. Forecast model coverage varies by variable and location; unavailable
# values may be returned as null or omitted by the service.
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "temperature_80m",
    "temperature_120m",
    "temperature_180m",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "terrestrial_radiation",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "terrestrial_radiation_instant",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "precipitation",
    "precipitation_probability",
    "rain",
    "showers",
    "snowfall",
    "snow_depth",
    "weather_code",
    "freezing_level_height",
    "visibility",
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm",
    "uv_index",
    "uv_index_clear_sky",
    "is_day",
    "sunshine_duration",
    "wet_bulb_temperature_2m",
    "precipitable_water",
    "vapour_pressure_deficit",
    "cape",
    "lifted_index",
    "convective_inhibition",
    "boundary_layer_height",
]

PRESSURE_LEVELS_HPA = (
    1000,
    975,
    950,
    925,
    900,
    850,
    800,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    70,
    50,
    30,
)
PRESSURE_LEVEL_VARIABLES = (
    f"{prefix}_{level}hPa"
    for prefix in (
        "temperature",
        "relative_humidity",
        "dew_point",
        "cloud_cover",
        "wind_speed",
        "wind_direction",
        "geopotential_height",
    )
    for level in PRESSURE_LEVELS_HPA
)
HOURLY_VARIABLES.extend(PRESSURE_LEVEL_VARIABLES)

CURRENT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "is_day",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

MINUTELY_15_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "snowfall_height",
    "freezing_level_height",
    "sunshine_duration",
    "weather_code",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_gusts_10m",
    "visibility",
    "cape",
    "lightning_potential",
    "is_day",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "terrestrial_radiation",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "terrestrial_radiation_instant",
]

DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_mean",
    "apparent_temperature_min",
    "uv_index_max",
    "uv_index_clear_sky_max",
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "moonrise",
    "moonset",
    "moon_phase",
    "rain_sum",
    "showers_sum",
    "snowfall_sum",
    "precipitation_sum",
    "precipitation_hours",
    "precipitation_probability_max",
    "precipitation_probability_mean",
    "precipitation_probability_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "cape_mean",
    "cape_max",
    "cape_min",
    "cloud_cover_mean",
    "cloud_cover_max",
    "cloud_cover_min",
    "dew_point_2m_mean",
    "dew_point_2m_max",
    "dew_point_2m_min",
    "et0_fao_evapotranspiration_sum",
    "growing_degree_days_base_0_limit_50",
    "leaf_wetness_probability_mean",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "snowfall_water_equivalent_sum",
    "pressure_msl_mean",
    "pressure_msl_max",
    "pressure_msl_min",
    "surface_pressure_mean",
    "surface_pressure_max",
    "surface_pressure_min",
    "updraft_max",
    "visibility_mean",
    "visibility_min",
    "visibility_max",
    "wind_gusts_10m_mean",
    "wind_speed_10m_mean",
    "wind_gusts_10m_min",
    "wind_speed_10m_min",
    "wet_bulb_temperature_2m_mean",
    "wet_bulb_temperature_2m_max",
    "wet_bulb_temperature_2m_min",
    "vapour_pressure_deficit_max",
]


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("数値を指定してください") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("0 より大きい有限値を指定してください")
    return result


def nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("数値を指定してください") from exc
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("0 以上の有限値を指定してください")
    return result


def latitude_value(value: str) -> float:
    number = _coordinate(value, "緯度")
    if not -90 <= number <= 90:
        raise argparse.ArgumentTypeError("緯度は -90 から 90 の範囲で指定してください")
    return number


def longitude_value(value: str) -> float:
    number = _coordinate(value, "経度")
    if not -180 <= number <= 180:
        raise argparse.ArgumentTypeError("経度は -180 から 180 の範囲で指定してください")
    return number


def _coordinate(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label}は数値で指定してください") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"{label}は有限値で指定してください")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="指定座標の Open-Meteo 予報データを全変数取得して JSON 保存します。"
    )
    parser.add_argument("--latitude", required=True, type=latitude_value, help="緯度 (WGS84, -90..90)")
    parser.add_argument("--longitude", required=True, type=longitude_value, help="経度 (WGS84, -180..180)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="保存先 (既定: data)")
    parser.add_argument("--forecast-days", type=int, choices=range(1, 17), default=16, metavar="1..16")
    parser.add_argument("--past-days", type=int, choices=range(0, 93), default=0, metavar="0..92")
    parser.add_argument(
        "--minutely-15-steps",
        type=int,
        default=96,
        metavar="N",
        help="15分値の取得ステップ数。96=24時間、0で15分値を無効化 (既定: 96)",
    )
    parser.add_argument("--timezone", default="auto", help="タイムゾーン名または auto (既定: auto)")
    parser.add_argument(
        "--wind-speed-unit",
        choices=("ms", "kmh", "mph", "kn"),
        default="ms",
        help="風速単位 (既定: ms、MPC計算向け)",
    )
    parser.add_argument("--temperature-unit", choices=("celsius", "fahrenheit"), default="celsius")
    parser.add_argument("--precipitation-unit", choices=("mm", "inch"), default="mm")
    parser.add_argument("--panel-tilt-degrees", type=float, default=0.0, help="GTI計算用パネル傾斜角")
    parser.add_argument(
        "--panel-azimuth-degrees",
        type=float,
        default=0.0,
        help="GTI計算用方位角 (0=南、-90=東、90=西、±180=北)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=nonnegative_float,
        default=0,
        metavar="SEC",
        help="指定すると繰り返し取得。0なら1回のみ (既定: 0)",
    )
    parser.add_argument("--timeout-seconds", type=positive_float, default=30, metavar="SEC")
    parser.add_argument("--retries", type=int, choices=range(0, 11), default=3, metavar="0..10")
    parser.add_argument("--no-history", action="store_true", help="履歴ファイルを作らず latest.json のみ更新")
    parser.add_argument("--quiet", action="store_true", help="成功時のログを抑制")
    args = parser.parse_args(argv)
    if args.minutely_15_steps < 0:
        parser.error("--minutely-15-steps は 0 以上で指定してください")
    if not math.isfinite(args.panel_tilt_degrees) or not 0 <= args.panel_tilt_degrees <= 90:
        parser.error("--panel-tilt-degrees は 0 から 90 の範囲で指定してください")
    if not math.isfinite(args.panel_azimuth_degrees) or not -180 <= args.panel_azimuth_degrees <= 180:
        parser.error("--panel-azimuth-degrees は -180 から 180 の範囲で指定してください")
    return args


def build_params(args: argparse.Namespace) -> dict[str, str]:
    params = {
        "latitude": str(args.latitude),
        "longitude": str(args.longitude),
        "current": ",".join(CURRENT_VARIABLES),
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
        "forecast_days": str(args.forecast_days),
        "past_days": str(args.past_days),
        "timezone": args.timezone,
        "temperature_unit": args.temperature_unit,
        "wind_speed_unit": args.wind_speed_unit,
        "precipitation_unit": args.precipitation_unit,
        "timeformat": "iso8601",
        "tilt": str(args.panel_tilt_degrees),
        "azimuth": str(args.panel_azimuth_degrees),
    }
    if args.minutely_15_steps:
        params["minutely_15"] = ",".join(MINUTELY_15_VARIABLES)
        params["forecast_minutely_15"] = str(args.minutely_15_steps)
    return params


def fetch_json(
    params: dict[str, str], timeout: float, retries: int
) -> tuple[dict[str, Any], str, dict[str, str], list[dict[str, str]]]:
    effective_params = params.copy()
    omitted_variables: list[dict[str, str]] = []
    attempt = 0
    while True:
        url = f"{API_URL}?{urlencode(effective_params)}"
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                status = response.status
            if status < 200 or status >= 300:
                raise RuntimeError(f"Open-Meteo が HTTP {status} を返しました")
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("Open-Meteo の応答が JSON object ではありません")
            if payload.get("error"):
                raise RuntimeError(f"Open-Meteo API エラー: {payload.get('reason', 'unknown error')}")
            return payload, url, effective_params, omitted_variables
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            invalid = re.search(
                r"invalid String value\s+([A-Za-z0-9_]+)\s+for key\s+(hourly|daily|current|minutely_15)",
                detail,
            )
            if exc.code == 400 and invalid:
                variable, key = invalid.groups()
                selected = effective_params.get(key, "").split(",")
                if variable in selected:
                    selected.remove(variable)
                    if selected and selected[0]:
                        effective_params[key] = ",".join(selected)
                    else:
                        effective_params.pop(key, None)
                    omitted_variables.append({"group": key, "variable": variable})
                    LOG.warning("API が未対応の変数 %s (%s) を除いて再取得します", variable, key)
                    continue
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= retries:
                raise RuntimeError(f"Open-Meteo HTTP {exc.code}: {detail[:1000]}") from exc
            wait_seconds = min(30.0, 2**attempt + random.uniform(0, 0.5))
            LOG.warning("HTTP %s。%.1f 秒後に再試行します", exc.code, wait_seconds)
            time.sleep(wait_seconds)
            attempt += 1
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Open-Meteo への接続に失敗しました: {exc}") from exc
            wait_seconds = min(30.0, 2**attempt + random.uniform(0, 0.5))
            LOG.warning("通信に失敗しました。%.1f 秒後に再試行します: %s", wait_seconds, exc)
            time.sleep(wait_seconds)
            attempt += 1


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(document, temp_file, ensure_ascii=False, indent=2, allow_nan=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        Path(temp_name).replace(path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def collect_once(args: argparse.Namespace, params: dict[str, str]) -> Path:
    payload, request_url, effective_params, omitted_variables = fetch_json(
        params, args.timeout_seconds, args.retries
    )
    retrieved_at = datetime.now(timezone.utc)
    document = {
        "schema_version": 1,
        "retrieved_at_utc": retrieved_at.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "requested_coordinate": {
            "latitude": args.latitude,
            "longitude": args.longitude,
        },
        "request": {
            "endpoint": API_URL,
            "parameters": params,
            "effective_parameters": effective_params,
            "omitted_unsupported_variables": omitted_variables,
            "url": request_url,
        },
        "api_response": payload,
    }

    output_dir = args.output_dir
    latest_path = output_dir / "latest.json"
    atomic_write_json(latest_path, document)
    if not args.no_history:
        timestamp = retrieved_at.strftime("%Y%m%dT%H%M%S.%fZ")
        history_path = output_dir / "history" / f"forecast_{timestamp}.json"
        atomic_write_json(history_path, document)
    return latest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    params = build_params(args)
    LOG.info("Open-Meteo 取得を開始: lat=%s, lon=%s", args.latitude, args.longitude)
    while True:
        started = time.monotonic()
        try:
            latest_path = collect_once(args, params)
            if not args.quiet:
                LOG.info("取得・保存完了: %s", latest_path.resolve())
        except KeyboardInterrupt:
            LOG.info("停止しました")
            return 0
        except Exception as exc:  # Keep a long-running collector alive on transient API errors.
            LOG.error("取得に失敗しました: %s", exc)
            if args.interval_seconds <= 0:
                return 1
        if args.interval_seconds <= 0:
            return 0
        remaining = args.interval_seconds - (time.monotonic() - started)
        if remaining > 0:
            try:
                time.sleep(remaining)
            except KeyboardInterrupt:
                LOG.info("停止しました")
                return 0


if __name__ == "__main__":
    sys.exit(main())
