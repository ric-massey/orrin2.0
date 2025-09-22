# observability/metrics.py
# Prometheus metrics for errors, heartbeat, reaper trips, and all watchdogs.

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# --- Error metrics ---
errors_total = Counter(
    "orrin_errors_total",
    "Count of errors by key and severity",
    ["key", "severity"]
)
error_rate_trips_total = Counter(
    "orrin_error_rate_trips_total",
    "Trip count for error rate limits",
    ["scope", "key"]  # scope=any|key, key="" for scope=any
)
error_threshold_trips_total = Counter(
    "orrin_error_threshold_trips_total",
    "Trip count for repeated-error thresholds",
    ["key", "severity"]
)

# --- Heartbeat metrics (kept as-is for existing tests) ---
hb_avg_period_ms = Gauge(
    "orrin_hb_avg_period_ms",
    "Current average heartbeat period (ms)"
)
hb_fast_streak = Gauge(
    "orrin_hb_fast_streak",
    "Consecutive checks below min period"
)
hb_slow_streak = Gauge(
    "orrin_hb_slow_streak",
    "Consecutive checks above max period"
)
hb_interval_ms = Histogram(
    "orrin_hb_interval_ms",
    "Distribution of heartbeat intervals (ms)",
    buckets=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 5_000, 10_000]
)

# --- Reaper trips (kills) ---
reaper_trips_total = Counter(
    "orrin_reaper_trips_total",
    "Number of times reaper triggered",
    ["reason"]
)

# --------------------------------------------------------------------
# New watchdog-family metrics (optional to use, safe to import everywhere)
# --------------------------------------------------------------------

# Liveness (by section)
liveness_missed_total = Counter(
    "orrin_liveness_missed_total",
    "Times a section missed its max_missed_cycles",
    ["section"]
)

# Lifespan (random hard cutoff)
lifespan_limit_cycles = Gauge(
    "orrin_lifespan_limit_cycles",
    "The secret cycle limit chosen at startup (if exposed)"
)
lifespan_cycles = Gauge(
    "orrin_lifespan_cycles",
    "Current total cycles (pulse)"
)

# Goals / No-goals guard
no_goals_idle_trips_total = Counter(
    "orrin_no_goals_idle_trips_total",
    "Trips due to no active/updated goals for too many cycles",
    []
)
retry_saturation_trips_total = Counter(
    "orrin_retry_saturation_trips_total",
    "Trips due to sustained retry rate above threshold",
    []
)
cb_open_too_long_trips_total = Counter(
    "orrin_cb_open_too_long_trips_total",
    "Trips due to a breaker staying OPEN too long",
    ["name"]
)
cb_many_open_trips_total = Counter(
    "orrin_cb_many_open_trips_total",
    "Trips due to too many distinct breakers OPEN in a window",
    []
)

# Memory / FD / CPU guard
rss_mb_gauge = Gauge(
    "orrin_rss_mb",
    "Process resident set size (MB)"
)
fd_pct_open_gauge = Gauge(
    "orrin_fd_pct_open",
    "Open file descriptors as fraction of limit (0..1)"
)
sock_pct_open_gauge = Gauge(
    "orrin_sock_pct_open",
    "Open sockets as fraction of limit (0..1)"
)
cpu_util_gauge = Gauge(
    "orrin_cpu_util",
    "CPU utilization (0..1)"
)
step_latency_ms_gauge = Gauge(
    "orrin_step_latency_ms",
    "Recent loop/step latency (ms)"
)
memory_leak_trips_total = Counter(
    "orrin_memory_leak_trips_total",
    "Trips due to memory slope exceeding threshold",
    []
)
fd_pressure_trips_total = Counter(
    "orrin_fd_pressure_trips_total",
    "Trips due to FD pressure sustained above threshold",
    []
)
socket_pressure_trips_total = Counter(
    "orrin_socket_pressure_trips_total",
    "Trips due to socket pressure sustained above threshold",
    []
)
cpu_starvation_trips_total = Counter(
    "orrin_cpu_starvation_trips_total",
    "Trips due to sustained high CPU with rising/high latency",
    []
)

# Repeat-loop guard
repeat_soft_breakers_open_total = Counter(
    "orrin_repeat_soft_breakers_open_total",
    "Soft trips (breaker opens) for repeat/ping-pong/retry saturation",
    ["scope"]  # scope = fp|retry
)
repeat_hard_trips_total = Counter(
    "orrin_repeat_hard_trips_total",
    "Hard trips for repeat/ping-pong/no-progress/retry escalation",
    ["reason"]  # reason tokens like repeat_same_call_loop|repeat_ping_pong_loop|no_progress_loop|retry_saturation
)

def serve_metrics(port: int = 9100):
    """Expose /metrics on http://localhost:<port>/metrics"""
    start_http_server(port)
