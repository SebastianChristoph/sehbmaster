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

  // Weather
  getWeatherData: (city: string, leadDays?: number, dateFrom?: string, dateTo?: string) =>
    get<WeatherDataPoint[]>("/weather/data", {
      city,
      model: "default",
      ...(leadDays !== undefined ? { lead_days: leadDays } : {}),
      ...(dateFrom ? { date_from: dateFrom } : {}),
      ...(dateTo ? { date_to: dateTo } : {}),
      limit: 5000,
    }),
  getWeatherLogs: (limit = 200) =>
    get<LogEntry[]>("/weather/logs", { limit }),
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

export interface WeatherDataPoint {
  id: number;
  target_date: string;
  lead_days: number;
  model: string;
  city: string;
  run_time: string;
  weather: string | null;
  temp_avg_c: number | null;
  temp_min_c: number | null;
  temp_max_c: number | null;
  wind_mps: number | null;
  rain_mm: number | null;
  created_at: string;
  rain_probability_pct: number | null;
}

export interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
}
