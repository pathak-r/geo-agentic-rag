const prefix = "/api";

async function j<T>(resPromise: Promise<Response>): Promise<T> {
  const res = await resPromise;
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json() as T;
}

export type Meta = {
  wells: string[];
  date_min: string;
  date_max: string;
  total_wells: number;
  production_rows: number;
  total_oil_sm3: number;
};

export function getMeta() {
  return j<Meta>(fetch(`${prefix}/meta`));
}

export function getHealth() {
  return j<{ ok: boolean; error: string | null }>(fetch(`${prefix}/health`));
}

export type FieldOilRow = {
  DATEPRD: string;
  WELL_NAME: string;
  BORE_OIL_VOL: number;
};

export function getFieldOilByWell(start?: string, end?: string) {
  const q = new URLSearchParams();
  if (start) q.set("start", start);
  if (end) q.set("end", end);
  return j<{ rows: FieldOilRow[] }>(
    fetch(`${prefix}/production/field-oil-by-well?${q}`)
  );
}

export function getProductionSummary(well?: string, start?: string, end?: string) {
  const q = new URLSearchParams();
  if (well && well !== "All Wells") q.set("well", well);
  if (start) q.set("start", start);
  if (end) q.set("end", end);
  return j<{ rows: Record<string, unknown>[] }>(
    fetch(`${prefix}/production/summary?${q}`)
  );
}

export type WellDetailRow = {
  DATEPRD: string;
  WELL_NAME: string;
  BORE_OIL_VOL: number;
  BORE_WAT_VOL: number;
  WATER_CUT_PCT: number;
  AVG_WHP_P: number;
  BORE_GAS_VOL: number;
};

export function getWellDetail(well: string, start?: string, end?: string) {
  const q = new URLSearchParams({ well });
  if (start) q.set("start", start);
  if (end) q.set("end", end);
  return j<{
    rows: WellDetailRow[];
    metrics: {
      total_oil_sm3: number;
      avg_water_cut_pct: number;
      production_days: number;
      avg_whp: number | null;
    };
  }>(fetch(`${prefix}/production/well-detail?${q}`));
}

export type AnomalyRow = Record<string, unknown>;

export function getAnomalies(well?: string) {
  const q = new URLSearchParams();
  if (well && well !== "All Wells") q.set("well", well);
  return j<{
    rows: AnomalyRow[];
    counts: Record<string, number>;
  }>(fetch(`${prefix}/anomalies?${q}`));
}

export type ChatMessage = { role: "user" | "assistant"; content: string };

export function postChat(message: string, history: ChatMessage[]) {
  return j<{ response: string; sources: { doc_type: string; excerpt: string }[] }>(
    fetch(`${prefix}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    })
  );
}
