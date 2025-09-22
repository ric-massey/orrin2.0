[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
![status](https://img.shields.io/badge/status-WIP-orange)

# Orrin 2.0 (WIP)

> ⚠️ This repo is evolving rapidly. APIs and files may change.

## Quickstart
```bash
# Python 3.11+
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt


# REAPER
The reaper .py is the kill switch for the main loop. It uses a watchdog to always run in the backround. Reaper checks for errors, heartbeat(loop consistancy), keeps it in a life span of a set number of loops, checks to see if the memory slope is rising too fast, makes sure the cpu isnt starved, etc. 
The reaper when one of the cercumstances above become true, shut down the program. 

# MEMORY
The memory/daemon.py is the always-on recall engine for the system. It tails every event via a write-ahead log, embeds and scores items (novelty/strength), deduplicates, and continuously compacts/promotes them from short-term buffers into long-term store for fast retrieval. It also extracts structured definitions into a lexicon, serves queries to the loop, and enforces safety budgets (rate limits, size caps, slope checks) so memory growth stays healthy. If ingestion stalls, corruption is detected, or backpressure crosses thresholds, Memory throttles itself and signals the main loop (or Reaper) to shut down cleanly.

# GOALS
The goals subsystem is Orrin 2.0’s planner/executor. A daemon (goals_daemon.py) scans NEW goals, expands them into READY steps, and schedules work via a policy (policy.py) that balances priority, deadlines, fairness, and optional locks. A runner (runner.py) executes steps through registered handlers (registry.py/handlers/*) with health and metrics taps, while a simple API/CLI (api.py, cli.py) lets you create/list/describe/update/cancel goals from code or terminal. State is persisted with a file store/WAL and snapshots (store.py, wal.py, snapshots.py), and triggers/health checks throttle or pause execution if queues, errors, or backpressure cross thresholds—so the main loop stays consistent and responsive.

# USER INTERFACE
The UI folder hosts the live dashboards for Orrin 2.0. It’s a pair of Vite/React apps (metrics-dashboard/ and memory-dashboard/) that poll your local endpoints (Prometheus-style /metrics and /memory/health) to visualize heartbeat consistency, CPU/RSS, FD/socket pressure, step latency, and trip counters (reaper/memory/cpu/etc.). It’s read-only monitoring: rolling time-series charts and quick stats so you can spot drift, pressure, or stalls and decide when to intervene or let Reaper act. Will be updated in the future. 


## License
This project is licensed under the Apache-2.0 License.  
See [LICENSE](./LICENSE) for details.

SPDX-License-Identifier: Apache-2.0