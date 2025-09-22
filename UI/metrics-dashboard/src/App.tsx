// UI/metrics-dashboard/src/App.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart
} from "recharts";

/** --- Prometheus text parser (very small, enough for our metrics) --- */
type Sample = { name: string; labels: Record<string,string>; value: number; };
function parseProm(text: string): Sample[] {
  const out: Sample[] = [];
  const lines = text.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    // examples:
    // orrin_hb_avg_period_ms 23.18
    // orrin_hb_interval_ms_bucket{le="20.0"} 59
    const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)$/);
    if (!m) continue;
    const name = m[1];
    const labelsStr = m[2];
    const val = Number(m[3]);
    const labels: Record<string,string> = {};
    if (labelsStr) {
      const inner = labelsStr.slice(1, -1); // remove {}
      if (inner.trim().length) {
        const parts = inner.match(/(\w+)="[^"]*"/g) || [];
        for (const p of parts) {
          const mm = p.match(/^(\w+)="(.*)"$/);
          if (mm) labels[mm[1]] = mm[2];
        }
      }
    }
    out.push({ name, labels, value: val });
  }
  return out;
}

/** Keep a rolling time-series for a few key metrics */
type Point = { t: number; v: number };
function useTimeseries(value: number | undefined, keepMs = 60_000, everyMs = 1000) {
  const [data, setData] = useState<Point[]>([]);
  useEffect(() => {
    const id = setInterval(() => {
      setData(d => {
        const now = Date.now();
        const v = (value ?? 0);
        const next = [...d, { t: now, v }];
        const cutoff = now - keepMs;
        while (next.length && next[0].t < cutoff) next.shift();
        return next;
      });
    }, everyMs);
    return () => clearInterval(id);
  }, [value, keepMs, everyMs]);
  return data;
}

/** Small helpers to pull a metric */
function getGauge(samples: Sample[], metric: string): number | undefined {
  const s = samples.find(s => s.name === metric);
  return s?.value;
}
function getCounter(samples: Sample[], metric: string, labels?: Record<string,string>): number | undefined {
  return samples.find(s =>
    s.name === metric &&
    (!labels || JSON.stringify(s.labels) === JSON.stringify(labels))
  )?.value;
}

/** NEW: Cycle bubble component (label: "Cycle count") */
function CycleBubble({ value }: { value: number | undefined }) {
  const display = value !== undefined ? value.toLocaleString() : "—";
  return (
    <div className="card" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div
        style={{
          width: 120,
          height: 120,
          borderRadius: 9999,
          border: "2px solid #1d2633",
          background: "#0f141b",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 8px 20px rgba(0,0,0,.25)"
        }}
        aria-label="Cycle count"
        title={value !== undefined ? `Cycle count: ${value}` : "Cycle count unavailable"}
      >
        <div style={{ textAlign: "center", lineHeight: 1.1 }}>
          <div className="muted" style={{ fontSize: 12 }}>Cycle count</div>
          <div style={{ fontSize: 22, fontWeight: 700 }}>{display}</div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    async function tick() {
      try {
        const res = await fetch("/metrics", { cache: "no-store" });
        const txt = await res.text();
        if (!alive) return;
        setSamples(parseProm(txt));
        setErr(null);
      } catch (e: any) {
        setErr(e?.message || "Failed to fetch /metrics");
      }
    }
    tick();
    const id = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // Pull interesting values
  const hbAvg  = useMemo(() => getGauge(samples, "orrin_hb_avg_period_ms"), [samples]);
  const hbFast = useMemo(() => getGauge(samples, "orrin_hb_fast_streak"), [samples]);
  const hbSlow = useMemo(() => getGauge(samples, "orrin_hb_slow_streak"), [samples]);

  const rssMb  = useMemo(() => getGauge(samples, "orrin_rss_mb"), [samples]);
  const fdPct  = useMemo(() => getGauge(samples, "orrin_fd_pct_open"), [samples]);
  const sockPct= useMemo(() => getGauge(samples, "orrin_sock_pct_open"), [samples]);
  const cpuUtil= useMemo(() => getGauge(samples, "orrin_cpu_util"), [samples]);
  const stepLat= useMemo(() => getGauge(samples, "orrin_step_latency_ms"), [samples]);

  const reaperTrips = useMemo(() => getCounter(samples, "orrin_reaper_trips_total"), [samples]);
  const memTrips    = useMemo(() => getCounter(samples, "orrin_memory_leak_trips_total"), [samples]);
  const fdTrips     = useMemo(() => getCounter(samples, "orrin_fd_pressure_trips_total"), [samples]);
  const sockTrips   = useMemo(() => getCounter(samples, "orrin_socket_pressure_trips_total"), [samples]);
  const cpuTrips    = useMemo(() => getCounter(samples, "orrin_cpu_starvation_trips_total"), [samples]);
  const goalsTrips  = useMemo(() => getCounter(samples, "orrin_no_goals_idle_trips_total"), [samples]);
  const retryTrips  = useMemo(() => getCounter(samples, "orrin_retry_saturation_trips_total"), [samples]);
  const cbManyTrips = useMemo(() => getCounter(samples, "orrin_cb_many_open_trips_total"), [samples]);

  // NEW: current cycle (prefer orrin_lifespan_cycles; fallback to orrin_cycle)
  const currentCycle = useMemo(() => {
    const n = getGauge(samples, "orrin_lifespan_cycles")
          ?? getGauge(samples, "orrin_cycle");
    return Number.isFinite(n) ? Math.floor(n as number) : undefined;
  }, [samples]);

  // time series
  const hbSeries  = useTimeseries(hbAvg,  120_000, 1000);
  const cpuSeries = useTimeseries(cpuUtil,120_000, 1000);
  const latSeries = useTimeseries(stepLat,120_000, 1000);

  const formatTs = (t: number) => new Date(t).toLocaleTimeString();

  return (
    <div className="wrap">
      <h1>Orrin Metrics</h1>
      {err && <div className="card" style={{borderColor:"#5f1d1d", background:"#1a1010"}}>Error: {err}</div>}

      <div className="grid" style={{marginBottom:16}}>
        <div className="card">
          <div className="title">Heartbeat avg period</div>
          <div className="stat">{hbAvg?.toFixed(2) ?? "—"} <span className="muted">ms</span></div>
          <div className="muted">fast streak: {hbFast ?? 0} • slow streak: {hbSlow ?? 0}</div>
        </div>

        <div className="card">
          <div className="title">RSS</div>
          <div className="stat">{rssMb?.toFixed(1) ?? "—"} <span className="muted">MB</span></div>
        </div>

        <div className="card">
          <div className="title">FD / Socket</div>
          <div className="stat">
            {(fdPct!=null ? Math.round(fdPct*100) : "—")}% <span className="muted">FD</span>
          </div>
          <div className="muted">Sockets: {sockPct!=null ? Math.round(sockPct*100) : "—"}%</div>
        </div>

        <div className="card">
          <div className="title">CPU util</div>
          <div className="stat">{cpuUtil!=null ? Math.round(cpuUtil*100) : "—"}%</div>
          <div className="muted">step latency: {stepLat?.toFixed(1) ?? "—"} ms</div>
        </div>

        {/* Trips card */}
        <div className="card">
          <div className="title">Trips</div>
          <div className="muted">reaper: {reaperTrips ?? 0} • mem: {memTrips ?? 0} • fd: {fdTrips ?? 0} • sock: {sockTrips ?? 0}</div>
          <div className="muted">cpu: {cpuTrips ?? 0} • no-goals: {goalsTrips ?? 0} • retry: {retryTrips ?? 0} • CB-many: {cbManyTrips ?? 0}</div>
        </div>

        {/* Bubble as its own grid item — sits to the right of Trips on wide screens */}
        <CycleBubble value={currentCycle} />
      </div>

      <div className="row" style={{marginBottom:16}}>
        <div className="card">
          <div className="title">Heartbeat period (ms)</div>
          <div style={{width:"100%", height:220}}>
            <ResponsiveContainer>
              <AreaChart data={hbSeries.map(p => ({ name: formatTs(p.t), v: p.v }))}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#7cc5ff" stopOpacity={0.6}/>
                    <stop offset="95%" stopColor="#7cc5ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#223044" strokeDasharray="3 3"/>
                <XAxis dataKey="name" tick={{fill:"#9fb3c8"}}/>
                <YAxis tick={{fill:"#9fb3c8"}}/>
                <Tooltip contentStyle={{background:"#0f141b", border:"1px solid #1d2633", color:"#e9eef3"}} />
                <Area type="monotone" dataKey="v" stroke="#7cc5ff" fill="url(#g1)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="title">CPU util (0..1)</div>
          <div style={{width:"100%", height:220}}>
            <ResponsiveContainer>
              <LineChart data={cpuSeries.map(p => ({ name: formatTs(p.t), v: p.v }))}>
                <CartesianGrid stroke="#223044" strokeDasharray="3 3"/>
                <XAxis dataKey="name" tick={{fill:"#9fb3c8"}}/>
                <YAxis domain={[0,1]} tick={{fill:"#9fb3c8"}}/>
                <Tooltip contentStyle={{background:"#0f141b", border:"1px solid #1d2633", color:"#e9eef3"}} />
                <Line type="monotone" dataKey="v" stroke="#9fe27a" dot={false}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="title">Step latency (ms)</div>
          <div style={{width:"100%", height:220}}>
            <ResponsiveContainer>
              <LineChart data={latSeries.map(p => ({ name: formatTs(p.t), v: p.v }))}>
                <CartesianGrid stroke="#223044" strokeDasharray="3 3"/>
                <XAxis dataKey="name" tick={{fill:"#9fb3c8"}}/>
                <YAxis tick={{fill:"#9fb3c8"}}/>
                <Tooltip contentStyle={{background:"#0f141b", border:"1px solid #1d2633", color:"#e9eef3"}} />
                <Line type="monotone" dataKey="v" stroke="#f2c078" dot={false}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="muted">
        Data source: <code>/metrics</code>. Keep your Python app running (<code>serve_metrics(port=9100)</code>).
      </div>
    </div>
  );
}
