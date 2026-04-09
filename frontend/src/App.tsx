import { useCallback, useEffect, useState } from "react";
import { getHealth, getMeta, type Meta } from "./api";
import { AnomaliesPanel } from "./components/AnomaliesPanel";
import { ChatPanel } from "./components/ChatPanel";
import { Dashboard } from "./components/Dashboard";

type Tab = "dash" | "chat" | "anom";

export default function App() {
  const [tab, setTab] = useState<Tab>("dash");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [well, setWell] = useState<string>("All Wells");
  const [start, setStart] = useState<string>("");
  const [end, setEnd] = useState<string>("");

  const load = useCallback(async () => {
    setErr(null);
    try {
      const h = await getHealth();
      if (!h.ok) {
        setErr(h.error || "Backend data not loaded.");
        setMeta(null);
        return;
      }
      const m = await getMeta();
      setMeta(m);
      setStart((s) => s || m.date_min);
      setEnd((e) => e || m.date_max);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to reach API");
      setMeta(null);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 shrink-0 border-r border-slate-800 bg-slate-900/80 p-4 flex flex-col gap-4">
        <div>
          <h1 className="text-lg font-bold text-sky-400">Geo-Agentic RAG</h1>
          <p className="text-xs text-slate-500">Volve field assistant</p>
        </div>

        {meta && (
          <>
            <label className="block text-xs font-medium text-slate-400">Well</label>
            <select
              value={well}
              onChange={(e) => setWell(e.target.value)}
              className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm"
            >
              <option>All Wells</option>
              {meta.wells.map((w) => (
                <option key={w} value={w}>
                  {w}
                </option>
              ))}
            </select>

            <label className="block text-xs font-medium text-slate-400">Start date</label>
            <input
              type="date"
              value={start}
              min={meta.date_min}
              max={meta.date_max}
              onChange={(e) => setStart(e.target.value)}
              className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm"
            />
            <label className="block text-xs font-medium text-slate-400">End date</label>
            <input
              type="date"
              value={end}
              min={meta.date_min}
              max={meta.date_max}
              onChange={(e) => setEnd(e.target.value)}
              className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm"
            />

            <div className="text-xs text-slate-500 space-y-1 pt-2 border-t border-slate-800">
              <div>Wells: {meta.total_wells}</div>
              <div>Days: {meta.production_rows.toLocaleString()}</div>
              <div>Oil Σ: {meta.total_oil_sm3.toLocaleString(undefined, { maximumFractionDigits: 0 })} Sm³</div>
            </div>
          </>
        )}

        <p className="text-[10px] text-slate-600 mt-auto leading-relaxed">
          Equinor Volve open data. Not financial or operational advice.
        </p>
      </aside>

      <main className="flex-1 flex flex-col min-w-0">
        <header className="border-b border-slate-800 px-6 py-3 flex items-center gap-2">
          {(["dash", "chat", "anom"] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setTab(k)}
              className={`px-3 py-1.5 rounded text-sm font-medium transition ${
                tab === k
                  ? "bg-sky-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {k === "dash" ? "Production" : k === "chat" ? "AI assistant" : "Anomalies"}
            </button>
          ))}
        </header>

        <div className="flex-1 overflow-auto p-6">
          {err && (
            <div className="rounded border border-amber-800 bg-amber-950/40 text-amber-200 px-4 py-3 text-sm mb-4">
              <strong>API / data:</strong> {err}
              <div className="text-xs mt-2 text-amber-300/80">
                Start backend from repo root:{" "}
                <code className="bg-black/30 px-1 rounded">uvicorn backend.main:app --reload --port 8000</code>
              </div>
            </div>
          )}

          {meta && tab === "dash" && (
            <Dashboard well={well} start={start} end={end} />
          )}
          {meta && tab === "chat" && <ChatPanel />}
          {meta && tab === "anom" && <AnomaliesPanel well={well} />}
        </div>
      </main>
    </div>
  );
}
