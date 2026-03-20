import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api, VergabeNotice, VergabeAlert, LogEntry } from "../api/client";
import { KPICard } from "../components/KPICard";
import { StatusBadge } from "../components/StatusBadge";
import { FileText, Euro, AlertTriangle, RefreshCw, Terminal, ChevronLeft, ChevronRight } from "lucide-react";

const SCRAPER_ID = "server-vergabewatch";
const PAGE_SIZE = 25;

function fmt(dt: string | null) {
  if (!dt) return "—";
  return new Date(dt).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", year: "2-digit" });
}

function fmtEur(val: number | null) {
  if (val == null) return "—";
  if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(1)} Mio €`;
  if (val >= 1_000) return `${(val / 1_000).toFixed(0)} T€`;
  return `${val.toFixed(0)} €`;
}

const ALERT_LABELS: Record<string, { label: string; color: string }> = {
  REPEAT_WINNER: { label: "Wiederholungsgewinner", color: "bg-rose-100 text-rose-700" },
  NEAR_THRESHOLD: { label: "Knapp unter Schwellenwert", color: "bg-amber-100 text-amber-700" },
};

// eForms procedure type codes → German label + description
const PROCEDURE_META: Record<string, { label: string; desc: string; color: string }> = {
  open:         { label: "Offen",             desc: "Offenes Verfahren – jeder Bieter kann ein Angebot abgeben",                         color: "bg-emerald-100 text-emerald-700" },
  restricted:   { label: "Nicht offen",        desc: "Nichtoffenes Verfahren – nur ausgewählte Bieter werden aufgefordert",               color: "bg-sky-100 text-sky-700" },
  "neg-w-call": { label: "Verhandlung (mit)",  desc: "Verhandlungsverfahren mit vorherigem Aufruf zum Wettbewerb",                       color: "bg-violet-100 text-violet-700" },
  "neg-wo-call":{ label: "Verhandlung (ohne)", desc: "Verhandlungsverfahren ohne vorherigen Aufruf – z.B. bei Dringlichkeit oder Alleinstellung", color: "bg-amber-100 text-amber-700" },
  "comp-tend":  { label: "Wettb. Dialog",      desc: "Wettbewerblicher Dialog – für besonders komplexe Aufträge",                        color: "bg-indigo-100 text-indigo-700" },
  "oth-single": { label: "Direktvergabe",      desc: "Direktvergabe an einen einzigen Anbieter",                                        color: "bg-rose-100 text-rose-700" },
  "oth-mult":   { label: "Direktvergabe (m.)", desc: "Direktvergabe nach Marktabfrage bei mehreren Anbietern",                           color: "bg-orange-100 text-orange-700" },
  innovation:   { label: "Innovation",         desc: "Innovationspartnerschaft – für Entwicklung neuer Lösungen",                        color: "bg-teal-100 text-teal-700" },
  __null__:     { label: "Unbekannt",          desc: "Verfahrensart nicht aus XML extrahierbar",                                        color: "bg-slate-100 text-slate-500" },
};

function ProcedureBadge({ code }: { code: string | null }) {
  const key = code ?? "__null__";
  const meta = PROCEDURE_META[key] ?? { label: key, desc: key, color: "bg-slate-100 text-slate-500" };
  return (
    <span
      title={meta.desc}
      className={`inline-block text-[10px] font-medium px-1.5 py-0.5 rounded cursor-default ${meta.color}`}
    >
      {meta.label}
    </span>
  );
}

export function Vergabe() {
  const [notices, setNotices]   = useState<VergabeNotice[]>([]);
  const [stats, setStats]       = useState<{ total_notices: number; total_value_eur: number; top_contractors: { name: string; count: number }[]; top_authorities: { name: string; count: number }[] } | null>(null);
  const [alerts, setAlerts]     = useState<VergabeAlert[]>([]);
  const [logs, setLogs]         = useState<LogEntry[]>([]);
  const [scraperStatus, setScraperStatus] = useState<{ raspberry: string; status: string; message: string | null } | null>(null);
  const [procedureCounts, setProcedureCounts] = useState<{ code: string; count: number }[]>([]);
  const [loading, setLoading]   = useState(true);
  const [page, setPage]         = useState(0);
  const [filterProc, setFilterProc] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    Promise.all([
      api.getVergabeNotices(PAGE_SIZE, page * PAGE_SIZE, filterProc ?? undefined),
      api.getVergabeStats(),
      api.getVergabeAlerts(50),
      api.getVergabeLogs(50),
      api.getStatus(),
      api.getVergabeProcedureTypes(),
    ]).then(([n, s, alrt, lg, statuses, pc]) => {
      setNotices(n);
      setStats(s);
      setAlerts(alrt);
      setLogs(lg);
      setScraperStatus(statuses.find(st => st.raspberry === SCRAPER_ID) ?? null);
      setProcedureCounts(pc);
    }).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [page, filterProc]);

  const handleFilterProc = (code: string | null) => {
    setFilterProc(code);
    setPage(0);
  };

  const runLogs = logs.filter(l =>
    l.message.includes("finished vergabewatch") || l.message.includes("SCRAPER ERROR")
  ).slice(0, 10);

  // When filter is active, total for pagination is the count for that filter bucket
  const filteredTotal = filterProc
    ? (procedureCounts.find(p => p.code === filterProc)?.count ?? 0)
    : (stats?.total_notices ?? 0);
  const totalPages = Math.ceil(filteredTotal / PAGE_SIZE) || 1;

  if (loading) return <div className="p-8 text-slate-400 text-sm">Lade...</div>;

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Vergabewatch</h1>
          <p className="text-sm text-slate-500 mt-0.5">EU-Ausschreibungs-Tracker – Zuschläge, Muster, Auffälligkeiten</p>
        </div>
        <button onClick={load} className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-700">
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Aktualisieren
        </button>
      </div>

      {scraperStatus && (
        <div className="mb-6 bg-white border border-slate-200 rounded-lg px-5 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Terminal size={15} className="text-slate-400" />
            <span className="text-sm font-medium text-slate-700">Scraper ({SCRAPER_ID})</span>
            <StatusBadge status={scraperStatus.status} />
          </div>
          <p className="text-xs text-slate-500 truncate max-w-xl">{scraperStatus.message ?? "—"}</p>
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard label="Notices gesamt" value={(stats?.total_notices ?? 0).toLocaleString()} icon={<FileText size={16} />} />
        <KPICard label="Gesamtvolumen" value={fmtEur(stats?.total_value_eur ?? null)} icon={<Euro size={16} />} accent="emerald" />
        <KPICard label="Alerts" value={alerts.length} icon={<AlertTriangle size={16} />} accent="rose" />
        <KPICard label="Wiederholungen" value={alerts.filter(a => a.alert_type === "REPEAT_WINNER").length} icon={<AlertTriangle size={16} />} accent="amber" />
      </div>

      {/* Charts */}
      {stats && (stats.top_contractors.length > 0 || stats.top_authorities.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {stats.top_contractors.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-lg p-5">
              <h3 className="text-sm font-semibold text-slate-700 mb-4">Top Auftragnehmer</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={stats.top_contractors} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 9 }} width={140}
                    tickFormatter={v => v.length > 20 ? v.slice(0, 20) + "…" : v} />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <Bar dataKey="count" fill="#6366f1" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {stats.top_authorities.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-lg p-5">
              <h3 className="text-sm font-semibold text-slate-700 mb-4">Top Vergabestellen</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={stats.top_authorities} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 9 }} width={140}
                    tickFormatter={v => v.length > 20 ? v.slice(0, 20) + "…" : v} />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <Bar dataKey="count" fill="#f59e0b" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Alerts */}
      {alerts.length > 0 && (
        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">Alerts</h2>
          <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Typ</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Auftragnehmer</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Behörde</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide w-32">Datum</th>
                </tr>
              </thead>
              <tbody>
                {alerts.slice(0, 20).map((a, i) => {
                  const meta = ALERT_LABELS[a.alert_type] ?? { label: a.alert_type, color: "bg-slate-100 text-slate-600" };
                  return (
                    <tr key={a.id} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                      <td className="px-4 py-2">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded ${meta.color}`}>{meta.label}</span>
                      </td>
                      <td className="px-4 py-2 text-slate-700 text-xs truncate max-w-[200px]">{a.contractor ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-500 text-xs truncate max-w-[200px]">{a.authority ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-400 text-xs whitespace-nowrap">{fmt(a.created_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Notices Tabelle */}
      <section className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide">
            Vergabebekanntmachungen ({(stats?.total_notices ?? 0).toLocaleString()})
          </h2>
        </div>

        {/* Verfahren-Filter */}
        <div className="flex flex-wrap gap-2 mb-3">
          <button
            onClick={() => handleFilterProc(null)}
            className={`text-xs px-3 py-1 rounded-full border transition-colors ${
              filterProc === null
                ? "bg-slate-800 text-white border-slate-800"
                : "bg-white text-slate-600 border-slate-200 hover:border-slate-400"
            }`}
          >
            Alle
          </button>
          {procedureCounts.map(({ code, count }) => {
            const meta = PROCEDURE_META[code] ?? { label: code, desc: code, color: "bg-slate-100 text-slate-500" };
            const active = filterProc === code;
            return (
              <button
                key={code}
                onClick={() => handleFilterProc(code)}
                title={meta.desc}
                className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                  active
                    ? "bg-slate-800 text-white border-slate-800"
                    : "bg-white text-slate-600 border-slate-200 hover:border-slate-400"
                }`}
              >
                {meta.label} <span className="opacity-60">({count})</span>
              </button>
            );
          })}
        </div>

        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Datum</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Vergabestelle</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Auftragnehmer</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">CPV</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Wert</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Verfahren</th>
                </tr>
              </thead>
              <tbody>
                {notices.map((n, i) => (
                  <tr key={n.publication_number} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                    <td className="px-4 py-2.5 text-slate-400 text-xs whitespace-nowrap">{fmt(n.published_date)}</td>
                    <td className="px-4 py-2.5 text-slate-700 text-xs max-w-[200px] truncate">
                      <a href={n.ted_url ?? "#"} target="_blank" rel="noreferrer"
                        className="hover:text-indigo-600">{n.contracting_authority ?? "—"}</a>
                    </td>
                    <td className="px-4 py-2.5 text-slate-600 text-xs max-w-[200px] truncate">{n.contractor_name ?? "—"}</td>
                    <td className="px-4 py-2.5 text-xs">
                      {n.cpv_code ? (
                        <span title={n.cpv_description ?? n.cpv_code} className="font-mono text-slate-400 cursor-default">
                          {n.cpv_code}
                          {n.cpv_description && (
                            <span className="font-sans ml-1.5 text-slate-400 text-[10px] not-italic">{n.cpv_description}</span>
                          )}
                        </span>
                      ) : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-slate-700 text-xs whitespace-nowrap font-medium">{fmtEur(n.contract_value_eur)}</td>
                    <td className="px-4 py-2.5">
                      <ProcedureBadge code={n.procedure_type} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-slate-100 bg-slate-50">
              <span className="text-xs text-slate-500">Seite {page + 1} von {totalPages}</span>
              <div className="flex items-center gap-1">
                <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                  className="p-1.5 rounded text-slate-500 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed">
                  <ChevronLeft size={14} />
                </button>
                <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}
                  className="p-1.5 rounded text-slate-500 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed">
                  <ChevronRight size={14} />
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Procedure-Legende */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">Verfahrensarten – Legende</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {Object.entries(PROCEDURE_META).filter(([k]) => k !== "__null__").map(([code, meta]) => (
            <div key={code} className="bg-white border border-slate-200 rounded-lg px-4 py-3 flex items-start gap-3">
              <span className={`mt-0.5 text-[10px] font-medium px-1.5 py-0.5 rounded shrink-0 ${meta.color}`}>{meta.label}</span>
              <div>
                <p className="text-[10px] font-mono text-slate-400 mb-0.5">{code}</p>
                <p className="text-xs text-slate-500">{meta.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Scraper-Läufe */}
      <section>
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">Letzte Scraper-Läufe</h2>
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          {runLogs.length === 0 ? (
            <div className="p-6 text-center text-slate-400 text-sm">Noch keine Läufe</div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide w-36">Zeit</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide w-20">Status</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Details</th>
                </tr>
              </thead>
              <tbody>
                {runLogs.map((l, i) => {
                  const isErr = l.message.includes("SCRAPER ERROR");
                  return (
                    <tr key={l.id} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                      <td className="px-4 py-2 text-slate-400 text-xs whitespace-nowrap">{fmt(l.timestamp)}</td>
                      <td className="px-4 py-2">
                        {isErr
                          ? <span className="inline-flex items-center gap-1 text-xs text-rose-600"><span className="w-1.5 h-1.5 rounded-full bg-rose-400" /> Fehler</span>
                          : <span className="inline-flex items-center gap-1 text-xs text-emerald-600"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> OK</span>}
                      </td>
                      <td className="px-4 py-2 text-slate-600 text-xs">{l.message}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  );
}
