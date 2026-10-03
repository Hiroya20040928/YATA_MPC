# YATA_MPC

YATA solar-car energy-management research project. This repository contains the source code, fitted model artifacts, API schemas, Grafana dashboard configuration, and engineering reports for the BWSC2025 shadow replay and BWSC2027 preparation.

The runtime is advisory-only. It does not send throttle, PWM, or motor commands. A hard-constraint-feasible plan is required before a speed advisory is returned; when the optimizer has no eligible solution, the API withholds the advisory. The model and its historical replay do not certify vehicle safety, guarantee race-time improvement, or prove superiority to a human driver.

## MPC structure

- Upper planner: finite speed-grid MILP over the remaining road-day route. It minimizes predicted travel time subject to a terminal energy reserve and the schedule deadline. It replans every 300 seconds and on forecast, route-progress, tracking, or energy triggers.
- Lower controller: one-dimensional move-blocking QP with a 20-second prediction horizon and 1-second update interval. It tracks the upper speed target subject to an inferred acceleration envelope and terminal-speed bounds.
- The upper plan is invalidated during a due replan. No speed advice is returned while solving or when the original hard constraints cannot be met.

This is a finite-grid, model-based advisory system. It does not prove recursive feasibility or robust closed-loop performance: actuator response is inferred from historical speed changes and product data, not calibrated from logged commands and measured motor response. Read the limitations and validation record before operational use.

## Requirements

- Python 3.12 recommended for the analysis/replay scripts.
- Python packages listed in [requirements.txt](requirements.txt).
- Network access for Open-Meteo forecast retrieval and map tiles.
- Grafana 13.2.3 and the signed Infinity datasource plugin for the local dashboard. The start script expects the Grafana archive under `%LOCALAPPDATA%\YATA_MPC`.

Install the Python dependencies in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the live input API

```powershell
python scripts/mpc_live_input_api.py
```

The service listens on `127.0.0.1:8125` by default. Its principal endpoints are:

- `POST /v1/telemetry` — newest vehicle state, nominally 1 Hz.
- `POST /v1/forecast/route` — route-distance by forecast-valid-time weather grid.
- `POST /v1/race-schedule` — active route, day-end distance, deadline, reserve, and stop schedule.
- `GET /v1/controller/input` — current interpolated weather, fitted power preview, and battery estimate.
- `GET /v1/controller/advice` — upper plan and lower-layer advisory, or HTTP 503 when inputs or constraints are not eligible.
- `GET /api/health` — service and planner status.

Payload definitions are in [schemas/](schemas/), and the complete telemetry contract is in [reports/telemetry_input_requirements_v1.md](reports/telemetry_input_requirements_v1.md). The bundled DEM route is a BWSC2025 reference route; provide the official route and schedule for the intended event. Do not connect the advisory to vehicle actuators without separate response calibration and safety validation.

## Run the local historical Grafana preview

```powershell
.\scripts\start_local_grafana.ps1
```

The preview is available at `http://127.0.0.1:3000`. It replays local BWSC2025 data through a read-only API; it is not a live vehicle dashboard. Raw telemetry and replay outputs are intentionally excluded from this repository, so supply the required local files under `data/` before starting the replay service. See [grafana/README.md](grafana/README.md).

To fetch weather for an arbitrary coordinate:

```powershell
python .\openmeteo_fetcher.py --latitude=-33.8688 --longitude=151.2093
```

## Project layout

```text
src/yata_mpc/     Telemetry ingestion, battery estimator, and controller components
scripts/          Model fitting, replay, benchmarking, local APIs, and Grafana setup
models/           Fitted operating parameters and product-derived response tables
schemas/          Versioned HTTP payload contracts
grafana/          Dashboard JSON and local datasource provisioning
reports/          Engineering decisions, validation reports, and operating requirements
data/input/route/ Bundled BWSC2025 DEM reference route only
```

The race telemetry, raw weather responses, derived replay tables and figures, user-provided PDFs/workbooks, and literature PDFs are not committed. Some historical replay and Grafana commands require those local inputs; their required paths and provenance are documented in the reports and scripts.

## Engineering reports

- [Hierarchical MPC architecture and program inventory](reports/hierarchical_mpc_architecture_and_program_inventory.md)
- [BWSC2025 as-of replay and constraint outcomes](reports/hierarchical_mpc_2025_replay.md)
- [MPC infeasibility and advisory policy](reports/mpc_infeasibility_and_advisory_policy.md)
- [Telemetry input requirements](reports/telemetry_input_requirements_v1.md)
- [Upper-layer MPC tuning and human comparison limits](reports/upper_mpc_tuning_and_human_comparison.md)
- [Lower-layer tuning and inferred actuator envelope](reports/mpc_tuning_and_actuator_inference.md)

The BWSC2025 replay is a model-based shadow evaluation with post-race fitted parameters. A reported solver optimum applies only to the stated finite candidate problem; it is not a global optimum over the real vehicle or continuous speed choices.
