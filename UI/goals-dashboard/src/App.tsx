// UI/goals-dashboard/src/App.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, LineChart, Line
} from "recharts";

type Priority = "LOW" | "NORMAL" | "HIGH" | "CRITICAL";
type Status = "NEW" | "READY" | "RUNNING" | "DONE" | "FAILED" | "BLOCKED" | "PAUSED";

type Progress = { percent: number; note?: string; updated_at?: string | null };
type Goal = {
  id: string; title: string; kind: string; priority: Priority; status: Status;
  created_at?: string; updated_at?: string; deadline?: string | null;
  spec?: Record<string, any>;
  progress?: Progress;
  tags?: string[] | null;
  parent_id?: string | null;
};

function usePoll<T>(url: string, ms: number) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ts, setTs] = useState<number>(0);
  useEffect(() => {
    let alive = true; let t: any;
    const load = async () => {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error(res.status + " " + res.statusText);
        const json = await res.json();
        if (alive) { setData(json); setErr(null); setTs(Date.now()); }
      } catch (e:any) {
        if (alive) setErr(String(e));
      } finally {
        t = setTimeout(load, ms);
      }
    };
    load();
    return () => { alive = false; clearTimeout(t); };
  }, [url, ms]);
  return { data, err, ts };
}

function getParam(name: string, fallback: string): string {
  const url = new URL(window.location.href);
  return url.searchParams.get(name) || fallback;
}

function fmtAgo(ms: number) {
  if (!ms) return "—";
  const s = Math.round(ms/1000);
  if (s < 60) return s + "s";
  const m = Math.floor(s/60);
  if (m < 60) return m + "m";
  const h = Math.floor(m/60);
  return h + "h";
}

function StatusCounts({ goals }: { goals: Goal[] }) {
  const counts = useMemo(() => {
    const c: Record<Status, number> = { NEW:0, READY:0, RUNNING:0, DONE:0, FAILED:0, BLOCKED:0, PAUSED:0 };
    for (const g of goals) c[g.status] = (c[g.status] || 0) + 1;
    return Object.entries(c).map(([k,v]) => ({ status: k, count: v }));
  }, [goals]);
  return (
    <div className="card" style={{ gridColumn: "span 6" }}>
      <h3 className="h">Status distribution</h3>
      <div style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={counts}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1d2633" />
            <XAxis dataKey="status" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PriorityCounts({ goals }: { goals: Goal[] }) {
  const counts = useMemo(() => {
    const c: Record<Priority, number> = { LOW:0, NORMAL:0, HIGH:0, CRITICAL:0 };
    for (const g of goals) c[g.priority] = (c[g.priority] || 0) + 1;
    return [
      { name: "LOW", v: c.LOW },
      { name: "NORMAL", v: c.NORMAL },
      { name: "HIGH", v: c.HIGH },
      { name: "CRITICAL", v: c.CRITICAL },
    ];
  }, [goals]);
  return (
    <div className="card" style={{ gridColumn: "span 6" }}>
      <h3 className="h">Priority distribution</h3>
      <div style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={counts}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1d2633" />
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Line type="monotone" dataKey="v" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function GoalsTable({ goals }: { goals: Goal[] }) {
  return (
    <div className="card" style={{ gridColumn: "span 12" }}>
      <h3 className="h">Goals</h3>
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Title</th><th>Kind</th><th>Status</th><th>Priority</th><th>Progress</th><th className="right">Updated</th>
          </tr>
        </thead>
        <tbody>
          {goals.map(g => (
            <tr key={g.id}>
              <td className="muted">{g.id.slice(0,8)}</td>
              <td>{g.title || <span className="muted">(untitled)</span>}</td>
              <td><span className="pill">{g.kind || "—"}</span></td>
              <td>{g.status}</td>
              <td>{g.priority}</td>
              <td>{g.progress ? (g.progress.percent ?? 0).toFixed(0) + "%" : "—"}</td>
              <td className="right">{g.updated_at ? new Date(g.updated_at).toLocaleString() : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  // Data source: defaults to /goals.json (serve this via CLI export), override with ?src=URL
  const src = getParam("src", "/goals.json");
  const { data, err, ts } = usePoll<Goal[]>(src, 2000);

  const goals = useMemo(() => (data || []).slice().sort((a,b) => {
    const ap = a.priority === "CRITICAL" ? 3 : a.priority === "HIGH" ? 2 : a.priority === "NORMAL" ? 1 : 0;
    const bp = b.priority === "CRITICAL" ? 3 : b.priority === "HIGH" ? 2 : b.priority === "NORMAL" ? 1 : 0;
    // sort by priority desc, then updated desc
    const t = (new Date(b.updated_at||0).getTime() - new Date(a.updated_at||0).getTime());
    return bp - ap || t;
  }), [data]);

  return (
    <div className="wrap">
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 12 }}>
        <h1 style={{ margin: 0 }}>Goals Dashboard</h1>
        <div className="muted">source: <code>{src}</code> • updated {fmtAgo(Date.now()-ts)} ago</div>
        {err && <div style={{ marginLeft: "auto", color: "#ff9c9c" }}>Error: {err}</div>}
      </div>

      <div className="grid">
        <StatusCounts goals={goals} />
        <PriorityCounts goals={goals} />
        <GoalsTable goals={goals} />
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="muted">
          Tip: export JSON via <code>python -m goals.cli list --json {">"} goals.json</code> and serve alongside this app, or point to any URL with <code>?src=</code>.
        </div>
      </div>
    </div>
  );
}
