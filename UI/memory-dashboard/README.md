# Orrin Memory Dashboard

A sibling UI to `metrics-dashboard`, focused on the Memory subsystem.

## What it shows
- **Daemon** status (running/stopped), store type
- **Store size** (items, bytes)
- **WAL** status (enabled/disabled), queue length, last flush lag
- A **bytes over time** mini-chart
- Raw JSON of whatever your backend returns from `/memory`

The UI is resilient: it looks for common field names like `items`, `store_items`, `bytes`, `store_bytes`, `daemon_alive`, `wal_queue`, etc.
Any additional fields from `/memory` are displayed in a Details list.

## Build

```bash
cd UI/memory-dashboard
npm install
npm run build
```

This produces `dist/` which you can serve from your Python `dashboard_server`.

## Backend: add a `/memory` endpoint

Update `start_dashboard_server` to accept a `memory_health_provider` callback and expose it at `GET /memory`.
