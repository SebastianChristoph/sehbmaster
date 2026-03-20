import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { KPICard } from "../components/KPICard";
import { StatusBadge } from "../components/StatusBadge";
import { Newspaper, Cloud, RefreshCw } from "lucide-react";

const projects = [
  { name: "Bildwatch",    path: "/bild",    icon: Newspaper, desc: "Bild.de Artikel-Monitoring"    },
  { name: "Weatherwatch", path: "/weather", icon: Cloud,     desc: "Wettervorhersage-Tracking"     },
];

export function Home() {
  const [statuses, setStatuses] = useState<{ raspberry: string; status: string; message: string | null }[]>([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    api.getStatus()
      .then(setStatuses)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const working = statuses.filter(s => s.status?.toLowerCase() === "working").length;
  const errors  = statuses.filter(s => s.status?.toLowerCase() === "error").length;

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">Übersicht aller Scraper-Projekte</p>
        </div>
        <button onClick={load} className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-700">
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Aktualisieren
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <KPICard label="Projekte" value={projects.length} icon={<Newspaper size={16} />} />
        <KPICard label="Scraper" value={statuses.length} icon={<Cloud size={16} />} accent="emerald" />
        <KPICard label="Aktiv" value={working} icon={<RefreshCw size={16} />} accent="emerald" />
        <KPICard label="Fehler" value={errors} icon={<RefreshCw size={16} />} accent={errors > 0 ? "rose" : "indigo"} />
      </div>

      <section className="mb-8">
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">Projekte</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {projects.map(({ name, path, icon: Icon, desc }) => (
            <Link
              key={path}
              to={path}
              className="bg-white border border-slate-200 rounded-lg p-5 hover:border-indigo-300 hover:shadow-sm transition-all group"
            >
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-indigo-50 rounded-md text-indigo-500 group-hover:bg-indigo-100 transition-colors">
                  <Icon size={16} />
                </div>
                <span className="font-semibold text-slate-800 text-sm">{name}</span>
              </div>
              <p className="text-xs text-slate-500">{desc}</p>
            </Link>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">Scraper Status</h2>
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-slate-400 text-sm">Lade...</div>
          ) : statuses.length === 0 ? (
            <div className="p-8 text-center text-slate-400 text-sm">Keine Daten</div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Scraper</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Status</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Nachricht</th>
                </tr>
              </thead>
              <tbody>
                {statuses.map((s, i) => (
                  <tr key={s.raspberry} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                    <td className="px-4 py-3 font-medium text-slate-700">{s.raspberry}</td>
                    <td className="px-4 py-3"><StatusBadge status={s.status} /></td>
                    <td className="px-4 py-3 text-slate-500 truncate max-w-xs">{s.message ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  );
}
