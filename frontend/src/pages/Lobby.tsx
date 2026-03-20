import { useEffect, useState } from "react";
import { api, LobbyEntry, LobbyChange, LobbyAlert, LogEntry } from "../api/client";
import { KPICard } from "../components/KPICard";
import { StatusBadge } from "../components/StatusBadge";
import { Users, AlertTriangle, RefreshCw, Terminal, GitCommit, ChevronLeft, ChevronRight } from "lucide-react";

const SCRAPER_ID = "server-lobbywatch";
const PAGE_SIZE = 25;

function fmt(dt: string | null) {
  if (!dt) return "—";
  return new Date(dt).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", year: "2-digit", hour: "2-digit", minute: "2-digit" });
}

const CHANGE_LABELS: Record<string, { label: string; color: string }> = {
  CODEX_VIOLATION:     { label: "Codex-Verstoß",      color: "text-rose-600 bg-rose-50" },
  CLIENT_CHANGE:       { label: "Neuer Auftraggeber",  color: "text-amber-700 bg-amber-50" },
  FINANCIAL_CHANGE:    { label: "Finanzen geändert",   color: "text-indigo-600 bg-indigo-50" },
  LEGISLATIVE_CHANGE:  { label: "Gesetzgebung",        color: "text-violet-600 bg-violet-50" },
  ACTIVE_CHANGE:       { label: "Status geändert",     color: "text-slate-600 bg-slate-100" },
  FIELD_CHANGE:        { label: "Feld geändert",       color: "text-slate-500 bg-slate-50" },
  NEW_ENTRY:           { label: "Neu registriert",     color: "text-emerald-600 bg-emerald-50" },
};

const ALERT_LABELS: Record<string, { label: string; color: string }> = {
  CODEX_VIOLATION:       { label: "Codex-Verstoß",       color: "bg-rose-100 text-rose-700" },
  FINANCIAL_NONDISCLOSURE: { label: "Keine Finanzangaben", color: "bg-amber-100 text-amber-700" },
  NEW_ENTRY:             { label: "Neuregistrierung",    color: "bg-emerald-100 text-emerald-700" },
  NEW_CLIENT:            { label: "Neuer Auftraggeber",  color: "bg-indigo-100 text-indigo-700" },
};

export function Lobby() {
  const [entries, setEntries]           = useState<LobbyEntry[]>([]);
  const [entriesTotal, setEntriesTotal] = useState(0);
  const [changes, setChanges]           = useState<LobbyChange[]>([]);
  const [alerts, setAlerts]             = useState<LobbyAlert[]>([]);
  const [logs, setLogs]                 = useState<LogEntry[]>([]);
  const [scraperStatus, setScraperStatus] = useState<{ raspberry: string; status: string; message: string | null } | null>(null);
  const [loading, setLoading]           = useState(true);
  const [entryPage, setEntryPage]       = useState(0);

  const load = () => {
    setLoading(true);
    Promise.all([
      api.getLobbyEntries(PAGE_SIZE, entryPage * PAGE_SIZE),
      api.getLobbyEntriesCount(),
      api.getLobbyChanges(50),
      api.getLobbyAlerts(50),
      api.getLobbyLogs(50),
      api.getStatus(),
    ]).then(([ents, cnt, chg, alrt, lg, statuses]) => {
      setEntries(ents);
      setEntriesTotal(cnt.count);
      setChanges(chg);
      setAlerts(alrt);
      setLogs(lg);
      setScraperStatus(statuses.find(s => s.raspberry === SCRAPER_ID) ?? null);
    }).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [entryPage]);

  const runLogs = logs.filter(l =>
    l.message.includes("finished lobbywatch") || l.message.includes("SCRAPER ERROR")
  ).slice(0, 10);

  const totalPages = Math.ceil(entriesTotal / PAGE_SIZE);

  if (loading) return <div className="p-8 text-slate-400 text-sm">Lade...</div>;

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Lobbywatch</h1>
          <p className="text-sm text-slate-500 mt-0.5">Lobbyregister-Tracking – Änderungen, Auftraggeber, Ausgaben</p>
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
        <KPICard label="Lobbyisten" value={entriesTotal.toLocaleString()} icon={<Users size={16} />} />
        <KPICard label="Änderungen" value={changes.length} icon={<GitCommit size={16} />} accent="indigo" />
        <KPICard label="Alerts" value={alerts.length} icon={<AlertTriangle size={16} />} accent="rose" />
        <KPICard label="Codex-Verstöße" value={alerts.filter(a => a.alert_type === "CODEX_VIOLATION").length} icon={<AlertTriangle size={16} />} accent="rose" />
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">Alerts</h2>
          <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Typ</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Nr.</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Beschreibung</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide w-36">Datum</th>
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
                      <td className="px-4 py-2 text-xs font-mono text-slate-500">{a.register_number}</td>
                      <td className="px-4 py-2 text-slate-600 text-xs truncate max-w-xs">{a.description ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-400 text-xs whitespace-nowrap">{fmt(a.created_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Letzte Änderungen */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">Letzte Änderungen</h2>
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          {changes.length === 0 ? (
            <div className="p-6 text-center text-slate-400 text-sm">Noch keine Änderungen aufgezeichnet</div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide w-28">Datum</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Typ</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Organisation</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Geänderte Felder</th>
                </tr>
              </thead>
              <tbody>
                {changes.map((c, i) => {
                  const meta = CHANGE_LABELS[c.change_type ?? ""] ?? { label: c.change_type ?? "?", color: "text-slate-500 bg-slate-50" };
                  const diffKeys = c.diff ? Object.keys(c.diff).join(", ") : "—";
                  return (
                    <tr key={c.id} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                      <td className="px-4 py-2 text-slate-400 text-xs whitespace-nowrap">{fmt(c.detected_at)}</td>
                      <td className="px-4 py-2">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded ${meta.color}`}>{meta.label}</span>
                      </td>
                      <td className="px-4 py-2 text-slate-700 text-xs font-medium">{c.notes ?? c.register_number}</td>
                      <td className="px-4 py-2 text-slate-400 text-xs font-mono truncate max-w-xs">{diffKeys}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </section>

      {/* Einträge */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
          Lobbyisten ({entriesTotal.toLocaleString()})
        </h2>
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50">
                <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Nr.</th>
                <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Organisation</th>
                <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Ausgaben (€)</th>
                <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Themen</th>
                <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">Letzte Änderung</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => (
                <tr key={e.register_number} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                  <td className="px-4 py-2 text-xs font-mono text-slate-400">{e.register_number}</td>
                  <td className="px-4 py-2 text-slate-700 font-medium text-xs max-w-xs truncate">
                    <span className="flex items-center gap-1.5">
                      {e.codex_violation && <span className="w-1.5 h-1.5 rounded-full bg-rose-400 shrink-0" title="Codex-Verstoß" />}
                      {e.refuse_financial_info && <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" title="Keine Finanzangaben" />}
                      {e.name ?? "—"}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-slate-500 text-xs whitespace-nowrap">
                    {e.financial_expenses_from != null
                      ? `${(e.financial_expenses_from / 1000).toFixed(0)}k – ${(e.financial_expenses_to! / 1000).toFixed(0)}k`
                      : e.refuse_financial_info ? <span className="text-amber-500">verweigert</span> : "—"}
                  </td>
                  <td className="px-4 py-2 text-slate-400 text-xs truncate max-w-[200px]">
                    {(e.fields_of_interest ?? []).slice(0, 3).join(", ") || "—"}
                  </td>
                  <td className="px-4 py-2 text-slate-400 text-xs whitespace-nowrap">{fmt(e.last_update_date)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-slate-100 bg-slate-50">
              <span className="text-xs text-slate-500">Seite {entryPage + 1} von {totalPages} ({entriesTotal.toLocaleString()} gesamt)</span>
              <div className="flex items-center gap-1">
                <button onClick={() => setEntryPage(p => Math.max(0, p - 1))} disabled={entryPage === 0}
                  className="p-1.5 rounded text-slate-500 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed">
                  <ChevronLeft size={14} />
                </button>
                <button onClick={() => setEntryPage(p => Math.min(totalPages - 1, p + 1))} disabled={entryPage >= totalPages - 1}
                  className="p-1.5 rounded text-slate-500 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed">
                  <ChevronRight size={14} />
                </button>
              </div>
            </div>
          )}
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
                        {isErr ? (
                          <span className="inline-flex items-center gap-1 text-xs text-rose-600"><span className="w-1.5 h-1.5 rounded-full bg-rose-400" /> Fehler</span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-xs text-emerald-600"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> OK</span>
                        )}
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
