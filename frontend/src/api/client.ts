const BASE = "/api";

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("admin_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function get<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)));
  }
  const res = await fetch(url.toString(), { headers: authHeaders() });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}

export const api = {
  // Auth
  login: (username: string, password: string) =>
    post<{ access_token: string; token_type: string }>("/auth/login", { username, password }),

  // Status
  getStatus: () => get<{ raspberry: string; status: string; message: string | null }[]>("/status"),

  // Info
  getInfo: () => get<{ schemas: string[] }>("/info"),

  // Bild
  getBildArticles: (limit = 500, offset = 0) =>
    get<BildArticle[]>("/bild/articles", { limit, offset }),
  getBildMetrics: (limit = 2000) =>
    get<BildMetric[]>("/bild/metrics", { limit }),
  getBildChartsHourly: (days = 60) =>
    get<{ snapshot_avg: HourlyPoint[]; new_avg: HourlyPoint[] }>("/bild/charts/hourly", { days }),
  getBildChartsCategoryCounts: () =>
    get<Record<string, number>>("/bild/charts/category_counts"),
  getBildChartsDailyConversions: (days = 90) =>
    get<{ day: string; count: number }[]>("/bild/charts/daily_conversions", { days }),
  getBildCorrections: () =>
    get<BildCorrection[]>("/bild/corrections"),
  getBildLogs: (limit = 200) =>
    get<LogEntry[]>("/bild/logs", { limit }),

  // Lobby
  getLobbyEntries: (limit = 25, offset = 0) =>
    get<LobbyEntry[]>("/lobby/entries", { limit, offset }),
  getLobbyEntriesCount: () =>
    get<{ count: number }>("/lobby/entries/count"),
  getLobbyChanges: (limit = 50) =>
    get<LobbyChange[]>("/lobby/changes", { limit }),
  getLobbyAlerts: (limit = 50) =>
    get<LobbyAlert[]>("/lobby/alerts", { limit }),
  getLobbyLogs: (limit = 100) =>
    get<LogEntry[]>("/lobby/logs", { limit }),

  // Vergabe
  getVergabeNotices: (limit = 25, offset = 0) =>
    get<VergabeNotice[]>("/vergabe/notices", { limit, offset }),
  getVergabeStats: () =>
    get<VergabeStats>("/vergabe/notices/stats"),
  getVergabeAlerts: (limit = 50) =>
    get<VergabeAlert[]>("/vergabe/alerts", { limit }),
  getVergabeLogs: (limit = 100) =>
    get<LogEntry[]>("/vergabe/logs", { limit }),
};

// ---------- Types ----------
export interface BildArticle {
  id: string;
  title: string;
  url: string;
  category: string | null;
  is_premium: boolean;
  converted: boolean;
  published: string | null;
  converted_time: string | null;
  converted_duration_hours: number | null;
}

export interface BildMetric {
  id: number;
  ts_hour: string;
  snapshot_total: number;
  snapshot_premium: number;
  snapshot_premium_pct: number;
  new_count: number;
  new_premium_count: number;
  created_at: string;
}

export interface HourlyPoint {
  hour: number;
  Premium: number;
  Nicht_Premium: number;
}

export interface BildCorrection {
  id: string;
  title: string;
  published: string;
  source_url: string;
  article_url: string | null;
  message: string | null;
  created_at: string;
}

export interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
}

export interface LobbyEntry {
  register_number: string;
  name: string | null;
  legal_form: string | null;
  first_publication_date: string | null;
  last_update_date: string | null;
  active: boolean;
  current_entry_id: number | null;
  financial_expenses_from: number | null;
  financial_expenses_to: number | null;
  refuse_financial_info: boolean;
  codex_violation: boolean;
  fields_of_interest: string[] | null;
  client_orgs: { name: string; country: string | null }[] | null;
  client_persons: { firstName: string; lastName: string }[] | null;
  legislative_projects: { name: string; printingNumber: string | null }[] | null;
  created_at: string;
  updated_at: string;
}

export interface LobbyChange {
  id: number;
  register_number: string;
  detected_at: string;
  old_entry_id: number | null;
  new_entry_id: number | null;
  change_type: string;
  diff: Record<string, { old: unknown; new: unknown }> | null;
  notes: string | null;
}

export interface LobbyAlert {
  id: number;
  register_number: string;
  alert_type: string;
  description: string | null;
  evidence: Record<string, unknown> | null;
  created_at: string;
}

export interface VergabeNotice {
  id: number;
  publication_number: string;
  notice_type: string | null;
  published_date: string | null;
  contracting_authority: string | null;
  contracting_country: string | null;
  contractor_name: string | null;
  contract_value_eur: number | null;
  cpv_code: string | null;
  cpv_description: string | null;
  procedure_type: string | null;
  description: string | null;
  ted_url: string | null;
  created_at: string;
}

export interface VergabeAlert {
  id: number;
  alert_type: string;
  contractor: string | null;
  authority: string | null;
  evidence: Record<string, unknown> | null;
  created_at: string;
}

export interface VergabeStats {
  total_notices: number;
  total_value_eur: number;
  top_contractors: { name: string; count: number }[];
  top_authorities: { name: string; count: number }[];
}
