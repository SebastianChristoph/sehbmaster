import { useEffect, useState } from "react";
import { api, GovIncident, GovArticle } from "../api/client";
import { KPICard } from "../components/KPICard";
import { Plane, AlertCircle, Eye, ExternalLink } from "lucide-react";

function fmt(dt: string | null) {
  if (!dt) return "—";
  return new Date(dt).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", year: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function Gov() {
  const [incidents, setIncidents] = useState<GovIncident[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [detail, setDetail] = useState<{ incident: GovIncident; articles: GovArticle[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [filter, setFilter] = useState<"all" | "seen" | "unseen">("all");

  useEffect(() => {
    api.getGovIncidents(undefined, 200)
      .then(setIncidents)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selected == null) return;
    setDetailLoading(true);
    api.getGovIncidentDetail(selected)
      .then(setDetail)
      .finally(() => setDetailLoading(false));
  }, [selected]);

  const total = incidents.length;
  const unseen = incidents.filter(i => !i.seen).length;

  const filtered = incidents.filter(i => {
    if (filter === "seen") return i.seen;
    if (filter === "unseen") return !i.seen;
    return true;
  });

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-slate-800">Gov Tracker</h1>
        <p className="text-sm text-slate-500 mt-0.5">Regierungsflieger & Government Incidents</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard label="Incidents gesamt" value={total} icon={<Plane size={16} />} />
        <KPICard label="Ungesehen" value={unseen} icon={<AlertCircle size={16} />} accent={unseen > 0 ? "rose" : "emerald"} />
        <KPICard label="Gesehen" value={total - unseen} icon={<Eye size={16} />} accent="emerald" />
        <KPICard label="Ausgewählt" value={selected != null ? `#${selected}` : "—"} icon={<Plane size={16} />} accent="indigo" />
      </div>

      <div className="flex gap-1 mb-4">
        {(["all", "unseen", "seen"] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              filter === f ? "bg-indigo-50 text-indigo-700 border border-indigo-200" : "text-slate-500 hover:text-slate-700 border border-transparent"
            }`}
          >
            {f === "all" ? "Alle" : f === "unseen" ? "Ungesehen" : "Gesehen"}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Incidents list */}
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-100 bg-slate-50">
            <h3 className="text-sm font-semibold text-slate-700">{filtered.length} Incidents</h3>
          </div>
          {loading ? (
            <div className="p-8 text-slate-400 text-sm text-center">Lade...</div>
          ) : (
            <div className="divide-y divide-slate-100 max-h-[600px] overflow-y-auto">
              {filtered.map(inc => (
                <button
                  key={inc.id}
                  onClick={() => setSelected(inc.id)}
                  className={`w-full text-left px-4 py-3 hover:bg-slate-50 transition-colors ${selected === inc.id ? "bg-indigo-50" : ""}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className={`text-sm font-medium leading-snug ${inc.seen ? "text-slate-500" : "text-slate-800"}`}>
                      {inc.headline}
                    </p>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {!inc.seen && <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />}
                      <span className="text-xs text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">
                        {inc.articles_count}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">{fmt(inc.occurred_at ?? inc.created_at)}</p>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Detail panel */}
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          {selected == null ? (
            <div className="flex items-center justify-center h-full min-h-[300px] text-slate-400 text-sm">
              Incident auswählen
            </div>
          ) : detailLoading ? (
            <div className="flex items-center justify-center h-full min-h-[300px] text-slate-400 text-sm">
              Lade...
            </div>
          ) : detail ? (
            <div>
              <div className="px-5 py-4 border-b border-slate-100">
                <h3 className="text-sm font-semibold text-slate-800 leading-snug">{detail.incident.headline}</h3>
                <p className="text-xs text-slate-400 mt-1">{fmt(detail.incident.occurred_at ?? detail.incident.created_at)}</p>
              </div>
              <div className="divide-y divide-slate-100 max-h-[520px] overflow-y-auto">
                {detail.articles.length === 0 ? (
                  <p className="px-5 py-4 text-sm text-slate-400">Keine Artikel</p>
                ) : detail.articles.map(a => (
                  <div key={a.id} className="px-5 py-3">
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-sm text-slate-700 font-medium leading-snug">{a.title}</p>
                      <a href={a.link} target="_blank" rel="noreferrer" className="text-slate-400 hover:text-indigo-500 shrink-0">
                        <ExternalLink size={13} />
                      </a>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-slate-400">{a.source}</span>
                      {a.published_at && (
                        <span className="text-xs text-slate-400">{fmt(a.published_at)}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
