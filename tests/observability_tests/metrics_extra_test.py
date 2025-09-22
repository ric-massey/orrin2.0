# tests/reaper_tests/metrics_extra_test.py
import observability.metrics as m

# ---------- small helpers (local to this file) ----------

def _counter_value(counter, labels: dict | None = None) -> float:
    total = 0.0
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                if labels is None:
                    total += sample.value
                elif sample.labels == labels:
                    total += sample.value
    return total

def _gauge_value(gauge, labels: dict | None = None) -> float:
    for metric in gauge.collect():
        for sample in metric.samples:
            if sample.name == gauge._name:
                if (labels is None and not sample.labels) or (labels == sample.labels):
                    return sample.value
    raise AssertionError("Gauge sample not found")

# ---------- Liveness / Lifespan ----------

def test_liveness_missed_total_increments_per_section_label():
    lbl_a = {"section": "planner"}
    lbl_b = {"section": "retrieval"}

    base_a = _counter_value(m.liveness_missed_total, lbl_a)
    base_b = _counter_value(m.liveness_missed_total, lbl_b)

    m.liveness_missed_total.labels(**lbl_a).inc()
    m.liveness_missed_total.labels(**lbl_b).inc(2)

    assert _counter_value(m.liveness_missed_total, lbl_a) == base_a + 1
    assert _counter_value(m.liveness_missed_total, lbl_b) == base_b + 2

def test_lifespan_gauges_set_and_read():
    m.lifespan_limit_cycles.set(27777)
    m.lifespan_cycles.set(12345)

    assert _gauge_value(m.lifespan_limit_cycles) == 27777
    assert _gauge_value(m.lifespan_cycles) == 12345

# ---------- No-Goals / Saturation ----------

def test_no_goals_and_retry_counters_increment():
    base_idle = _counter_value(m.no_goals_idle_trips_total)
    base_retry = _counter_value(m.retry_saturation_trips_total)

    m.no_goals_idle_trips_total.inc()
    m.retry_saturation_trips_total.inc(3)

    assert _counter_value(m.no_goals_idle_trips_total) == base_idle + 1
    assert _counter_value(m.retry_saturation_trips_total) == base_retry + 3

def test_circuit_breaker_counters_increment_with_labels():
    lbl_long = {"name": "db_primary"}
    base_long = _counter_value(m.cb_open_too_long_trips_total, lbl_long)
    m.cb_open_too_long_trips_total.labels(**lbl_long).inc()
    assert _counter_value(m.cb_open_too_long_trips_total, lbl_long) == base_long + 1

    base_many = _counter_value(m.cb_many_open_trips_total)
    m.cb_many_open_trips_total.inc(2)
    assert _counter_value(m.cb_many_open_trips_total) == base_many + 2

# ---------- Memory / FD / CPU ----------

def test_resource_gauges_set_and_read():
    m.rss_mb_gauge.set(512.5)
    m.fd_pct_open_gauge.set(0.72)
    m.sock_pct_open_gauge.set(0.55)
    m.cpu_util_gauge.set(0.97)
    m.step_latency_ms_gauge.set(42.0)

    assert _gauge_value(m.rss_mb_gauge) == 512.5
    assert _gauge_value(m.fd_pct_open_gauge) == 0.72
    assert _gauge_value(m.sock_pct_open_gauge) == 0.55
    assert _gauge_value(m.cpu_util_gauge) == 0.97
    assert _gauge_value(m.step_latency_ms_gauge) == 42.0

def test_resource_trip_counters_increment():
    base_mem = _counter_value(m.memory_leak_trips_total)
    base_fd  = _counter_value(m.fd_pressure_trips_total)
    base_sk  = _counter_value(m.socket_pressure_trips_total)
    base_cpu = _counter_value(m.cpu_starvation_trips_total)

    m.memory_leak_trips_total.inc()
    m.fd_pressure_trips_total.inc(2)
    m.socket_pressure_trips_total.inc(3)
    m.cpu_starvation_trips_total.inc()

    assert _counter_value(m.memory_leak_trips_total) == base_mem + 1
    assert _counter_value(m.fd_pressure_trips_total) == base_fd + 2
    assert _counter_value(m.socket_pressure_trips_total) == base_sk + 3
    assert _counter_value(m.cpu_starvation_trips_total) == base_cpu + 1

# ---------- Repeat-loop guard ----------

def test_repeat_soft_and_hard_trip_counters_increment_with_labels():
    # soft
    lbl_soft_fp = {"scope": "fp"}
    lbl_soft_retry = {"scope": "retry"}
    base_soft_fp = _counter_value(m.repeat_soft_breakers_open_total, lbl_soft_fp)
    base_soft_retry = _counter_value(m.repeat_soft_breakers_open_total, lbl_soft_retry)

    m.repeat_soft_breakers_open_total.labels(**lbl_soft_fp).inc()
    m.repeat_soft_breakers_open_total.labels(**lbl_soft_retry).inc(4)

    assert _counter_value(m.repeat_soft_breakers_open_total, lbl_soft_fp) == base_soft_fp + 1
    assert _counter_value(m.repeat_soft_breakers_open_total, lbl_soft_retry) == base_soft_retry + 4

    # hard
    lbl_h1 = {"reason": "repeat_same_call_loop"}
    lbl_h2 = {"reason": "retry_saturation"}
    base_h1 = _counter_value(m.repeat_hard_trips_total, lbl_h1)
    base_h2 = _counter_value(m.repeat_hard_trips_total, lbl_h2)

    m.repeat_hard_trips_total.labels(**lbl_h1).inc(2)
    m.repeat_hard_trips_total.labels(**lbl_h2).inc()

    assert _counter_value(m.repeat_hard_trips_total, lbl_h1) == base_h1 + 2
    assert _counter_value(m.repeat_hard_trips_total, lbl_h2) == base_h2 + 1
