import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { api, BildCorrection, BildMetric } from "../api/client";
import { KPICard } from "../components/KPICard";
import { FileText, Lock, TrendingUp, RefreshCw } from "lucide-react";

function fmt(dt: string | null) {
  if (!dt) return "—";
  return new Date(dt).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function Bild() {
  const [metrics, setMetrics] = useState<BildMetric[]>([]);
  const [categoryCounts, setCategoryCounts] = useState<Record<string, number>>({});
  const [dailyConversions, setDailyConversions] = useState<{ day: string; count: number }[]>([]);
  const [corrections, setCorrections] = useState<BildCorrection[]>([]);
  const [hourly, setHourly] = useState<{
    snapshot_avg: { hour: number; Premium: number; Nicht_Premium: number }[];
    new_avg: { hour: number; Premium: number; Nicht_Premium: number }[];
  }>({ snapshot_avg: [], new_avg: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getBildMetrics(2000),
      api.getBildChartsCategoryCounts(),
      api.getBildChartsDailyConversions(90),
      api.getBildCorrections(),
      api.getBildChartsHourly(60),
    ]).then(([m, cats, daily, corr, h]) => {
      setMetrics(m);
      setCategoryCounts(cats);
      setDailyConversions(daily.slice(-60));
      setCorrections(corr);
      setHourly(h);
    }).finally(() => setLoading(false));
  }, []);

  const latest = metrics[metrics.length - 1];
  const totalArticles = latest?.snapshot_total ?? 0;
  const premiumCount = latest?.snapshot_premium ?? 0;
  const premiumPct = latest?.snapshot_premium_pct ?? 0;
  const corrCount = corrections.length;

  const categoryData = Object.entries(categoryCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([name, value]) => ({ name: name.length > 18 ? name.slice(0, 18) + "…" : name, value }));

  const metricsTimeline = metrics.slice(-48).map(m => ({
    time: new Date(m.ts_hour).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", hour: "2-digit" }),
    Total: m.snapshot_total,
    Premium: m.snapshot_premium,
  }));

  if (loading) return <div className="p-8 text-slate-400 text-sm">Lade...</div>;

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-slate-800">Bildwatch</h1>
        <p className="text-sm text-slate-500 mt-0.5">Bild.de Artikel-Monitoring & Paywall-Tracking</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard label="Artikel gesamt" value={totalArticles.toLocaleString()} icon={<FileText size={16} />} />
        <KPICard label="Premium" value={premiumCount.toLocaleString()} icon={<Lock size={16} />} accent="amber" />
        <KPICard label="Premium %" value={`${premiumPct.toFixed(1)} %`} icon={<TrendingUp size={16} />} accent="amber" />
        <KPICard label="Corrections" value={corrCount} icon={<RefreshCw size={16} />} accent="rose" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Artikel-Entwicklung (letzte 48 Messpunkte)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={metricsTimeline}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} interval={7} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="Total" stroke="#6366f1" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="Premium" stroke="#f59e0b" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Ø Stündliche Verteilung – Neu (60 Tage)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hourly.new_avg}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} tickFormatter={h => `${h}h`} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Premium" fill="#f59e0b" stackId="a" />
              <Bar dataKey="Nicht_Premium" fill="#6366f1" stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Kategorien</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={categoryData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={120} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Bar dataKey="value" fill="#6366f1" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Conversions pro Tag (90 Tage)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={dailyConversions}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="day" tick={{ fontSize: 10 }} interval={13} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <section>
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
          Corrections ({corrections.length})
        </h2>
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Titel</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Veröffentlicht</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Nachricht</th>
                </tr>
              </thead>
              <tbody>
                {corrections.slice(0, 100).map((c, i) => (
                  <tr key={c.id} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                    <td className="px-4 py-2.5 text-slate-700 max-w-xs truncate">
                      {c.article_url ? (
                        <a href={c.article_url} target="_blank" rel="noreferrer" className="hover:text-indigo-600">{c.title}</a>
                      ) : c.title}
                    </td>
                    <td className="px-4 py-2.5 text-slate-500 whitespace-nowrap">{fmt(c.published)}</td>
                    <td className="px-4 py-2.5 text-slate-500 max-w-xs truncate">{c.message ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
