import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, AreaChart, Area
} from "recharts";

type Dict = Record<string, any>;
type Health = Dict;

async function fetchJSON(path: string): Promise<any> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json();
}

function bytes(n?: number) {
  if (typeof n !== "number" || !isFinite(n)) return "—";
  const units = ["B","KB","MB","GB","TB"];
  let u = 0, v = n;
  while (v >= 1024 && u < units.length-1) { v/=1024; u++; }
  return `${v.toFixed(v<10?2:1)} ${units[u]}`;
}

function usePoll<T>(path: string, everyMs = 2000) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const j = await fetchJSON(path);
        if (!alive) return;
        setData(j);
        setErr(null);
      } catch (e:any) {
        if (!alive) return;
        setErr(e?.message || String(e));
      }
    };
    tick();
    const id = setInterval(tick, everyMs);
    return () => { alive = false; clearInterval(id); };
  }, [path, everyMs]);
  return { data, err };
}

type Point = { t: number; v: number };
function useSeries(value: number | undefined, keepMs = 2*60_000, everyMs = 1000) {
  const [data, setData] = useState<Point[]>([]);
  useEffect(() => {
    const id = setInterval(() => {
      setData(d => {
        const now = Date.now();
        const v = value ?? 0;
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

export default function App() {
  // Expect your Python server to expose GET /memory returning a JSON health dict
  const { data: health, err } = usePoll<Health>("/memory", 2000);

  // try common fields with fallbacks
  const daemonAlive = useMemo(() => Boolean(health?.daemon_alive ?? health?.daemon ?? health?.alive), [health]);
  const storeType   = useMemo(() => (health?.store_type ?? health?.store ?? "—") as string, [health]);
  const items       = useMemo(() => (health?.items ?? health?.store_items ?? health?.count) as number | undefined, [health]);
  const bytesUsed   = useMemo(() => (health?.bytes ?? health?.store_bytes ?? health?.mem_bytes) as number | undefined, [health]);

  const walEnabled  = useMemo(() => Boolean(health?.wal_enabled ?? health?.wal?.enabled), [health]);
  const walQueue    = useMemo(() => (health?.wal_queue ?? health?.wal?.queue ?? health?.wal?.pending) as number | undefined, [health]);
  const walLag      = useMemo(() => (health?.wal_lag_s ?? health?.wal?.lag_s ?? health?.wal_last_flush_s_ago) as number | undefined, [health]);

  const bytesSeries = useSeries(typeof bytesUsed === "number" ? bytesUsed : undefined, 5*60_000, 2000);
  const formatTs = (t:number) => new Date(t).toLocaleTimeString();

  // Prepare a generic detail list from the health object
  const detailEntries = useMemo(() => {
    if (!health) return [];
    const ignore = new Set(["store_type","store","items","store_items","bytes","store_bytes","mem_bytes","daemon_alive","daemon","alive","wal_enabled","wal_queue","wal","wal_lag_s","wal_last_flush_s_ago"]);
    return Object.entries(health)
      .filter(([k]) => !ignore.has(k))
      .map(([k,v]) => [k, typeof v === "number" ? v : (typeof v === "object" ? JSON.stringify(v) : String(v))]);
  }, [health]);

  return (
    <div className="wrap">
      <h1>Orrin Memory</h1>

      {err && <div className="card" style={{borderColor:"#5f1d1d", background:"#1a1010"}}>Error: {err}</div>}

      <div className="grid" style={{marginBottom:16}}>
        <div className="card">
          <div className="title">Daemon</div>
          <div className="stat">
            <span className={daemonAlive ? "ok" : "err"}>{daemonAlive ? "Running" : "Stopped"}</span>
          </div>
          <div className="muted">Store: <span className="badge">{storeType}</span></div>
        </div>

        <div className="card">
          <div className="title">Store Size</div>
          <div className="stat">{items ?? "—"} <span className="muted">items</span></div>
          <div className="muted">{bytes(bytesUsed)}</div>
        </div>

        <div className="card">
          <div className="title">WAL</div>
          <div className="stat">{walEnabled ? "Enabled" : "Disabled"}</div>
          <div className="muted">queue: {walQueue ?? "—"} • last flush: {walLag!=null ? `${walLag}s ago` : "—"}</div>
        </div>

        <div className="card">
          <div className="title">Raw keys present</div>
          <div className="muted">health contains {health ? Object.keys(health).length : 0} keys</div>
          <details style={{marginTop:8}}>
            <summary>Show JSON</summary>
            <pre style={{whiteSpace:"pre-wrap"}}>{health ? JSON.stringify(health, null, 2) : "—"}</pre>
          </details>
        </div>
      </div>

      <div className="row" style={{marginBottom:16}}>
        <div className="card">
          <div className="title">Bytes over time</div>
          <div style={{width:"100%", height:220}}>
            <ResponsiveContainer>
              <AreaChart data={bytesSeries.map(p => ({ name: formatTs(p.t), v: p.v }))}>
                <defs>
                  <linearGradient id="gmem" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#7cc5ff" stopOpacity={0.6}/>
                    <stop offset="95%" stopColor="#7cc5ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#223044" strokeDasharray="3 3"/>
                <XAxis dataKey="name" tick={{fill:"#9fb3c8"}}/>
                <YAxis tick={{fill:"#9fb3c8"}}/>
                <Tooltip contentStyle={{background:"#0f141b", border:"1px solid #1d2633", color:"#e9eef3"}} />
                <Area type="monotone" dataKey="v" stroke="#7cc5ff" fill="url(#gmem)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="title">Details</div>
          {detailEntries.length ? (
            <ul style={{margin:0, paddingLeft:18}}>
              {detailEntries.map(([k,v]) => (
                <li key={k}><span className="muted">{k}:</span> <span>{String(v)}</span></li>
              ))}
            </ul>
          ) : (
            <div className="muted">No extra fields in /memory yet.</div>
          )}
        </div>
      </div>

      <div className="muted">
        Data source: <code>/memory</code>. Serve this UI from your Python process and expose a memory health endpoint.
      </div>
    </div>
  );
}
