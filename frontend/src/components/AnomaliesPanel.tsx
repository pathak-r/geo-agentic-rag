import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { getAnomalies, type AnomalyRow } from "../api";

type Props = { well: string };

export function AnomaliesPanel({ well }: Props) {
  const [rows, setRows] = useState<AnomalyRow[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let c = false;
    (async () => {
      setErr(null);
      try {
        const r = await getAnomalies(well);
        if (!c) {
          setRows(r.rows);
          setCounts(r.counts || {});
        }
      } catch (e) {
        if (!c) setErr(e instanceof Error ? e.message : "Failed");
      }
    })();
    return () => {
      c = true;
    };
  }, [well]);

  const typeBars = Object.entries(
    rows.reduce<Record<string, number>>((acc, row) => {
      const t = String(row.ANOMALY_TYPE ?? "?");
      acc[t] = (acc[t] || 0) + 1;
      return acc;
    }, {})
  ).map(([name, value]) => ({ name, value }));

  const scatterData = rows.map((row) => ({
    x: String(row.DATEPRD ?? ""),
    y: Number(row.VALUE) || 0,
    z: String(row.ANOMALY_TYPE ?? ""),
    well: String(row.WELL_NAME ?? ""),
  }));

  if (err) return <p className="text-red-400 text-sm">{err}</p>;

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-semibold text-slate-200">Anomaly detection</h2>
      <p className="text-sm text-slate-500">
        Uses rolling z-scores on production data. Filter well from the sidebar ({well}).
      </p>

      <div className="flex flex-wrap gap-4">
        {["Critical", "High", "Medium"].map((sev) => (
          <div key={sev} className="rounded-lg border border-slate-800 bg-slate-900/50 px-4 py-2">
            <div className="text-xs text-slate-500">{sev}</div>
            <div className="text-xl font-semibold text-slate-100">{counts[sev] ?? 0}</div>
          </div>
        ))}
      </div>

      {typeBars.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-slate-400 mb-2">By type</h3>
          <div className="h-[280px] w-full max-w-3xl">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={typeBars} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#64748b" />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 10 }} stroke="#64748b" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                <Bar dataKey="value" fill="#38bdf8" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {scatterData.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-slate-400 mb-2">Timeline (value)</h3>
          <div className="h-[360px] w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 8, bottom: 40, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="x" name="Date" tick={{ fontSize: 8 }} stroke="#64748b" angle={-35} textAnchor="end" height={60} />
                <YAxis dataKey="y" name="Value" stroke="#64748b" />
                <ZAxis dataKey="z" name="Type" range={[40, 40]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) =>
                    active && payload?.[0] ? (
                      <div className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs">
                        <div>{String(payload[0].payload.well)}</div>
                        <div>{payload[0].payload.z}</div>
                        <div>
                          {payload[0].payload.x}: {payload[0].payload.y}
                        </div>
                      </div>
                    ) : null
                  }
                />
                <Scatter data={scatterData} fill="#f472b6" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {rows.length === 0 ? (
        <p className="text-slate-500 text-sm">No anomalies in the current selection.</p>
      ) : (
        <div className="overflow-x-auto rounded border border-slate-800">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-900 text-slate-400">
              <tr>
                {["DATEPRD", "WELL_NAME", "ANOMALY_TYPE", "METRIC", "VALUE", "SEVERITY"].map((h) => (
                  <th key={h} className="px-2 py-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 500).map((row, i) => (
                <tr key={i} className="border-t border-slate-800">
                  {["DATEPRD", "WELL_NAME", "ANOMALY_TYPE", "METRIC", "VALUE", "SEVERITY"].map((k) => (
                    <td key={k} className="px-2 py-1.5 text-slate-300">
                      {row[k] === undefined || row[k] === null ? "" : String(row[k])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > 500 && (
            <p className="text-slate-600 text-xs p-2">Showing first 500 of {rows.length} rows.</p>
          )}
        </div>
      )}
    </div>
  );
}
