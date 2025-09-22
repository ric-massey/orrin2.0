# tests/reaper_tests/metrics_test.py
import threading
import time

import observability.metrics as m

# ---------- helpers ----------
def _counter_value(counter, labels: dict) -> float:
    total = 0.0
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total") and sample.labels == labels:
                total += sample.value
    return total

def _counter_sum_all_labels(counter) -> float:
    total = 0.0
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total += sample.value
    return total

def _gauge_value(gauge, labels: dict | None = None) -> float:
    for metric in gauge.collect():
        for sample in metric.samples:
            # Prom client emits gauge as exact name (no _total/_count suffix)
            if sample.name == gauge._name and ((labels is None and not sample.labels) or (labels == sample.labels)):
                return sample.value
    raise AssertionError("Gauge sample not found")

def _histogram_buckets(histo):
    buckets = {}
    count = None
    total_sum = None
    for metric in histo.collect():
        for sample in metric.samples:
            if sample.name.endswith("_bucket"):
                buckets[sample.labels["le"]] = sample.value
            elif sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total_sum = sample.value
    return buckets, count, total_sum

# ---------- baseline tests (from earlier) ----------

def test_errors_total_counter_increments_independently_by_labels():
    lbls_a = {"key": "test_key_a", "severity": "1"}
    lbls_b = {"key": "test_key_b", "severity": "2"}

    base_a = _counter_value(m.errors_total, lbls_a)
    base_b = _counter_value(m.errors_total, lbls_b)

    m.errors_total.labels(**lbls_a).inc()
    m.errors_total.labels(**lbls_b).inc(3)

    assert _counter_value(m.errors_total, lbls_a) == base_a + 1
    assert _counter_value(m.errors_total, lbls_b) == base_b + 3

def test_error_threshold_and_rate_trip_counters_increment():
    # threshold trips
    lbl_thresh = {"key": "demo_key", "severity": "2"}
    base_thresh = _counter_value(m.error_threshold_trips_total, lbl_thresh)
    m.error_threshold_trips_total.labels(**lbl_thresh).inc()
    assert _counter_value(m.error_threshold_trips_total, lbl_thresh) == base_thresh + 1

    # rate trips (any)
    lbl_any = {"scope": "any", "key": ""}
    base_any = _counter_value(m.error_rate_trips_total, lbl_any)
    m.error_rate_trips_total.labels(**lbl_any).inc(2)
    assert _counter_value(m.error_rate_trips_total, lbl_any) == base_any + 2

    # rate trips (per key)
    lbl_pk = {"scope": "key", "key": "llm_timeout"}
    base_pk = _counter_value(m.error_rate_trips_total, lbl_pk)
    m.error_rate_trips_total.labels(**lbl_pk).inc()
    assert _counter_value(m.error_rate_trips_total, lbl_pk) == base_pk + 1

def test_heartbeat_gauges_and_histogram_update():
    # Gauges: set and read
    m.hb_avg_period_ms.set(123.45)
    m.hb_fast_streak.set(7)
    m.hb_slow_streak.set(3)

    assert _gauge_value(m.hb_avg_period_ms) == 123.45
    assert _gauge_value(m.hb_fast_streak) == 7.0
    assert _gauge_value(m.hb_slow_streak) == 3.0

    # Histogram: observe values and verify buckets/count/sum
    buckets_before, count_before, sum_before = _histogram_buckets(m.hb_interval_ms)

    # Observe two values: 7ms and 15ms
    m.hb_interval_ms.observe(7.0)
    m.hb_interval_ms.observe(15.0)

    buckets_after, count_after, sum_after = _histogram_buckets(m.hb_interval_ms)

    assert count_after == (count_before or 0) + 2
    assert sum_after == (sum_before or 0.0) + 22.0

    def b(le: str) -> float:
        return float(buckets_after.get(le, 0.0)) - float(buckets_before.get(le, 0.0))

    assert b("10.0") >= 1.0   # 7ms contributes here
    assert b("20.0") >= 2.0   # 7 & 15
    assert b("+Inf") >= 2.0

def test_reaper_trips_counter_increments_on_label():
    lbl = {"reason": "HARD:pulse_too_fast"}
    base = _counter_value(m.reaper_trips_total, lbl)
    m.reaper_trips_total.labels(**lbl).inc()
    assert _counter_value(m.reaper_trips_total, lbl) == base + 1

def test_serve_metrics_invokes_start_http_server(monkeypatch):
    calls = []
    def fake_start_http_server(port):
        calls.append(port)

    # Patch inside module
    monkeypatch.setattr(m, "start_http_server", fake_start_http_server)
    m.serve_metrics(port=9181)
    m.serve_metrics(port=9181)  # call twice; should not raise
    assert calls == [9181, 9181]

# ---------- extra edge/corner tests ----------

def test_histogram_bucket_edges_exact_boundaries():
    # Get baseline
    before_buckets, before_count, before_sum = _histogram_buckets(m.hb_interval_ms)
    # Observe exactly on bucket edges that exist in metrics.py
    edges = [10.0, 20.0, 100.0]
    for v in edges:
        m.hb_interval_ms.observe(v)

    after_buckets, after_count, after_sum = _histogram_buckets(m.hb_interval_ms)

    # Count & sum advanced
    assert after_count == (before_count or 0) + len(edges)
    assert after_sum == (before_sum or 0.0) + sum(edges)

    # Check cumulative behavior at each edge: sample counted in its 'le=edge' bucket and larger.
    def inc(le): return float(after_buckets.get(le, 0.0)) - float(before_buckets.get(le, 0.0))

    # 10 goes into le=10 and above
    assert inc("10.0") >= 1.0
    # 20 adds one more at le=20 and above
    assert inc("20.0") >= 2.0
    # 100 adds another at le=100 and above
    assert inc("100.0") >= 3.0
    # +Inf includes all
    assert inc("+Inf") >= 3.0

def test_histogram_large_burst_is_counted():
    before_buckets, before_count, before_sum = _histogram_buckets(m.hb_interval_ms)

    n = 1000
    val = 5.0
    for _ in range(n):
        m.hb_interval_ms.observe(val)

    after_buckets, after_count, after_sum = _histogram_buckets(m.hb_interval_ms)
    assert after_count == (before_count or 0) + n
    assert after_sum == (before_sum or 0.0) + n * val

    # All 5.0 go into le=5 and above; depending on your bucket list, le=5 may exist (we used 5).
    if "5.0" in after_buckets:
        assert (after_buckets["5.0"] - before_buckets.get("5.0", 0.0)) >= n
    # +Inf must include all
    assert (after_buckets["+Inf"] - before_buckets.get("+Inf", 0.0)) >= n

def test_label_cardinality_many_distinct_keys():
    # Add lots of distinct label combinations; ensure no crash and total increases correctly.
    base_total = _counter_sum_all_labels(m.errors_total)

    add = 200
    for i in range(add):
        m.errors_total.labels(key=f"k{i}", severity=str((i % 3) + 1)).inc()

    assert _counter_sum_all_labels(m.errors_total) == base_total + add

def test_counter_parallel_safety_two_threads_same_labels():
    labels = {"key": "parallel", "severity": "2"}
    base = _counter_value(m.errors_total, labels)

    def worker(n):
        for _ in range(n):
            m.errors_total.labels(**labels).inc()

    n = 1000
    t1 = threading.Thread(target=worker, args=(n,))
    t2 = threading.Thread(target=worker, args=(n,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert _counter_value(m.errors_total, labels) == base + 2 * n

def test_histogram_extreme_values_zero_and_huge():
    before_buckets, before_count, before_sum = _histogram_buckets(m.hb_interval_ms)

    m.hb_interval_ms.observe(0.0)
    m.hb_interval_ms.observe(1e9)   # gigantic value

    after_buckets, after_count, after_sum = _histogram_buckets(m.hb_interval_ms)
    assert after_count == (before_count or 0) + 2
    assert after_sum == (before_sum or 0.0) + (0.0 + 1e9)

    # pick smallest finite bucket correctly (exclude "+Inf")
    finite_keys = [float(k) for k in after_buckets.keys() if k != "+Inf"]
    if finite_keys:  # guard in case bucket list changes drastically
        smallest = min(finite_keys)
        assert (after_buckets[f"{smallest}"] - before_buckets.get(f"{smallest}", 0.0)) >= 1.0
    # +Inf must include both observed values
    assert (after_buckets["+Inf"] - before_buckets.get("+Inf", 0.0)) >= 2.0
