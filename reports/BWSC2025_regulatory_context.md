# BWSC2025 rules reviewed for log interpretation and MPC constraints

The complete official BWSC2025 Regulations, Release Version 2.0 (70 pages; published 2024-10-28), and Team Manager's Guide, Version 1.0 (41 pages), were extracted and read in full. Officially located Regulation Bulletins 1–5 were also reviewed. Regulation text is source context, not a command to this assistant or a substitute for the user's project instructions.

## Operating constraints relevant to a future cruise controller

- The event uses ACST (UTC+09:30); the reconstructed 2025 rows are timestamped in Australia/Darwin local time, which shares that offset during the event.
- The road-day schedule has an 08:00 start and 17:00 finish; public-road and authority requirements apply, and cars must not be driven after sunset.
- Regulations identify failure to maintain the stated 60 km/h minimum speed on open road, missing a control stop, or arriving after close as conditions that can end a Challenger's competition. The objective therefore cannot optimize speed without event-eligibility constraints.
- Mandatory Control Stops have a 30-minute minimum stop; the arriving driver has a short permitted reconfiguration period, and team access is restricted for the first 25 minutes.
- Roadside stopping is restricted to emergencies/traffic. A vehicle that leaves the road after mechanical failure needs the Clerk of the Course's permission before rejoining.
- Cruise-control/automatic-driving behavior must disengage on manual brake input or switch-off and cannot resume automatically; the speed recommender should remain advisory unless a compliant vehicle interface is specified.
- For Challenger, the 2025 solar collector limit is 6.000 m² and the battery energy limit is 11 MJ. These are 2025 class constraints, not assumed BWSC2027 values.
- Bulletin 5 made telemetry upload optional in 2025 and described a 10-second suggested format. The supplied ZP telemetry is at five-second intervals and is retained at that grain.

## Source links

- [BWSC2025 Event Regulations, Version 2.0](https://assets.worldsolarchallenge.org/app/uploads/2024/12/04154642/3297_2025_bwsc_regulations_release_v20_published_28_october_2024.pdf)
- [BWSC2025 Team Manager's Guide, Version 1.0](https://assets.worldsolarchallenge.org/app/uploads/2024/10/04152314/3279_2025_bwsc_team_managers_guide_release_v10_published_05062024_compressed_1.pdf)
- [BWSC2025 Regulation Bulletins](https://worldsolarchallenge.org/2025-event/regulations)

This file records 2025-only interpretation for the reconstructed logs. BWSC2027 Regulations V1.0 and the 2027 Team Manager's Guide V1.0 were published on 7 May 2026; their MPC-relevant clauses are summarized separately in `BWSC2027_regulatory_context.md`. Update that 2027 layer when later bulletins, Team Notices, and the official 2027 route notes are issued.
