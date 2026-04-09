import { useEffect, useMemo, useState, type ReactElement } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getFieldOilByWell,
  getProductionSummary,
  getWellDetail,
  type FieldOilRow,
  type WellDetailRow,
} from "../api";

const WELL_COLORS = [
  "#38bdf8",
  "#a78bfa",
  "#f472b6",
  "#34d399",
  "#fbbf24",
  "#fb923c",
  "#94a3b8",
];

type Props = { well: string; start: string; end: string };

function pivotField(rows: FieldOilRow[]) {
  const dates = [...new Set(rows.map((r) => r.DATEPRD))].sort();
  const wells = [...new Set(rows.map((r) => r.WELL_NAME))].sort();
  const data = dates.map((d) => {
    const point: Record<string, string | number> = { date: d };
    for (const w of wells) {
      const r = rows.find((x) => x.DATEPRD === d && x.WELL_NAME === w);
      point[w] = r?.BORE_OIL_VOL ?? 0;
    }
    return point;
  });
  return { wells, data };
}

export function Dashboard({ well, start, end }: Props) {
  const [fieldRows, setFieldRows] = useState<FieldOilRow[]>([]);
  const [summaryRows, setSummaryRows] = useState<Record<string, unknown>[]>([]);
  const [detailRows, setDetailRows] = useState<WellDetailRow[]>([]);
  const [metrics, setMetrics] = useState<{
    total_oil_sm3: number;
    avg_water_cut_pct: number;
    production_days: number;
    avg_whp: number | null;
  } | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  useEffect(() => {
    let cancel = false;
    (async () => {
      setLoadErr(null);
      try {
        if (well === "All Wells") {
          const fo = await getFieldOilByWell(start, end);
          const su = await getProductionSummary(undefined, start, end);
          if (!cancel) {
            setFieldRows(fo.rows);
            setSummaryRows(su.rows);
            setDetailRows([]);
            setMetrics(null);
          }
        } else {
          const d = await getWellDetail(well, start, end);
          if (!cancel) {
            setFieldRows([]);
            setSummaryRows([]);
            setDetailRows(d.rows);
            setMetrics(d.metrics);
          }
        }
      } catch (e) {
        if (!cancel) setLoadErr(e instanceof Error ? e.message : "Load failed");
      }
    })();
    return () => {
      cancel = true;
    };
  }, [well, start, end]);

  const pivoted = useMemo(() => pivotField(fieldRows), [fieldRows]);

  if (loadErr) {
    return <p className="text-red-400 text-sm">{loadErr}</p>;
  }

  if (well === "All Wells") {
    return (
      <div className="space-y-8">
        <h2 className="text-xl font-semibold text-slate-200">Field oil by well (Sm³)</h2>
        <div className="h-[420px] w-full min-w-0">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={pivoted.data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#64748b" />
              <YAxis tick={{ fontSize: 10 }} stroke="#64748b" />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
              <Legend />
              {pivoted.wells.map((w, i) => (
                <Area
                  key={w}
                  type="monotone"
                  dataKey={w}
                  stackId="1"
                  stroke={WELL_COLORS[i % WELL_COLORS.length]}
                  fill={WELL_COLORS[i % WELL_COLORS.length]}
                  fillOpacity={0.35}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <h3 className="text-lg font-medium text-slate-300">Well summary</h3>
        <div className="overflow-x-auto rounded border border-slate-800">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-900 text-slate-400">
              <tr>
                {summaryRows[0]
                  ? Object.keys(summaryRows[0]).map((k) => (
                      <th key={k} className="px-3 py-2 font-medium">
                        {k}
                      </th>
                    ))
                  : null}
              </tr>
            </thead>
            <tbody>
              {summaryRows.map((row, i) => (
                <tr key={i} className="border-t border-slate-800 hover:bg-slate-900/50">
                  {Object.values(row).map((v, j) => (
                    <td key={j} className="px-3 py-2 text-slate-300">
                      {v === null || v === undefined ? "" : String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  const detailChart = detailRows.map((r) => ({
    date: r.DATEPRD,
    oil: r.BORE_OIL_VOL,
    water: r.BORE_WAT_VOL,
    wc: r.WATER_CUT_PCT,
    whp: r.AVG_WHP_P > 0 ? r.AVG_WHP_P : null,
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-200">Well: {well}</h2>
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Metric label="Total oil (Sm³)" value={metrics.total_oil_sm3.toLocaleString(undefined, { maximumFractionDigits: 0 })} />
          <Metric label="Avg water cut %" value={metrics.avg_water_cut_pct.toFixed(1)} />
          <Metric label="Producing days" value={String(metrics.production_days)} />
          <Metric
            label="Avg WHP"
            value={metrics.avg_whp != null ? metrics.avg_whp.toFixed(1) : "—"}
          />
        </div>
      )}

      <ChartBlock title="Oil & water (Sm³)">
        <LineChart data={detailChart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} stroke="#64748b" />
          <YAxis tick={{ fontSize: 9 }} stroke="#64748b" />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
          <Legend />
          <Line type="monotone" dataKey="oil" name="Oil" stroke="#38bdf8" dot={false} strokeWidth={1} />
          <Line type="monotone" dataKey="water" name="Water" stroke="#f472b6" dot={false} strokeWidth={1} />
        </LineChart>
      </ChartBlock>

      <ChartBlock title="Water cut %">
        <LineChart data={detailChart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} stroke="#64748b" />
          <YAxis tick={{ fontSize: 9 }} stroke="#64748b" />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
          <Line type="monotone" dataKey="wc" name="WC %" stroke="#fbbf24" dot={false} strokeWidth={1} />
        </LineChart>
      </ChartBlock>

      <ChartBlock title="Wellhead pressure">
        <LineChart data={detailChart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} stroke="#64748b" />
          <YAxis tick={{ fontSize: 9 }} stroke="#64748b" />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
          <Line type="monotone" dataKey="whp" name="WHP" stroke="#fb923c" dot={false} strokeWidth={1} connectNulls />
        </LineChart>
      </ChartBlock>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}

function ChartBlock({ title, children }: { title: string; children: ReactElement }) {
  return (
    <div>
      <h3 className="text-sm font-medium text-slate-400 mb-2">{title}</h3>
      <div className="h-[280px] w-full min-w-0">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
}
