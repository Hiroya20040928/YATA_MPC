# YATA MPC local Grafana preview

This project starts Grafana and a read-only HTTP API on loopback:

- Grafana: http://127.0.0.1:3000
- Replay API health: http://127.0.0.1:8123/api/health
- Dashboard UID: yata-bwsc2025-local

The default view replays BWSC2025 logs at 1x from row 160 (the first sustained road-speed segment). Grafana refreshes at 5 seconds, matching the source log cadence. The replay CSV and official-route reconstruction stay on this computer. The basemap uses OpenStreetMap tiles and therefore requires network access.

Run scripts/start_local_grafana.ps1 to start both local processes; it prints the dashboard URL when ready. Open that URL in a browser. Run scripts/stop_local_grafana.ps1 to stop only those processes.

The display is a historical shadow replay. Actual telemetry is shown separately from the MPC proposal. No vehicle actuator, throttle, or PWM interface is connected. The estimated battery energy is based on a 3011 Wh design-capacity prior and the project model; it is not a BMS-measured SOC.
