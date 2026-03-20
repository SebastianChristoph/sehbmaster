import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { api, WeatherDataPoint } from "../api/client";
import { KPICard } from "../components/KPICard";
import { Cloud, Thermometer, Droplets, Wind } from "lucide-react";

const CITIES = ["Hamburg", "Berlin", "München"];

function fmt(dt: string | null) {
  if (!dt) return "—";
  return new Date(dt).toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit", year: "2-digit" });
}

export function Weather() {
  const [city, setCity] = useState(CITIES[0]);
  const [data, setData] = useState<WeatherDataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getWeatherData(city, 0)
      .then(d => setData(d.sort((a, b) => a.target_date.localeCompare(b.target_date))))
      .finally(() => setLoading(false));
  }, [city]);

  const latest = data[data.length - 1];
  const cities = CITIES;

  const tempData = data.slice(-60).map(d => ({
    date: d.target_date,
    "Min °C": d.temp_min_c,
    "Avg °C": d.temp_avg_c,
    "Max °C": d.temp_max_c,
  }));

  const rainData = data.slice(-60).map(d => ({
    date: d.target_date,
    "Regen mm": d.rain_mm,
    "Regen %": d.rain_probability_pct,
  }));

  return (
    <div className="p-8">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Weatherwatch</h1>
          <p className="text-sm text-slate-500 mt-0.5">Wettervorhersage-Genauigkeits-Tracking</p>
        </div>
        <div className="flex gap-1 bg-white border border-slate-200 rounded-lg p-1">
          {cities.map(c => (
            <button
              key={c}
              onClick={() => setCity(c)}
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                city === c ? "bg-indigo-50 text-indigo-700" : "text-slate-500 hover:text-slate-700"
              }`}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard label="Datenpunkte" value={data.length.toLocaleString()} icon={<Cloud size={16} />} />
        <KPICard
          label="Letzte Messung"
          value={fmt(latest?.target_date ?? null)}
          icon={<Thermometer size={16} />}
          accent="emerald"
        />
        <KPICard
          label="Ø Temperatur"
          value={latest?.temp_avg_c != null ? `${latest.temp_avg_c.toFixed(1)} °C` : "—"}
          icon={<Thermometer size={16} />}
          accent="amber"
        />
        <KPICard
          label="Regenwahrsch."
          value={latest?.rain_probability_pct != null ? `${latest.rain_probability_pct.toFixed(0)} %` : "—"}
          icon={<Droplets size={16} />}
          accent="indigo"
        />
      </div>

      {loading ? (
        <div className="p-8 text-slate-400 text-sm">Lade...</div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          <div className="bg-white border border-slate-200 rounded-lg p-5">
            <h3 className="text-sm font-semibold text-slate-700 mb-4">Temperaturverlauf – {city} (Ist-Werte, 60 Tage)</h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={tempData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={9} />
                <YAxis tick={{ fontSize: 10 }} unit=" °C" />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="Max °C" stroke="#f59e0b" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="Avg °C" stroke="#6366f1" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="Min °C" stroke="#94a3b8" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-5">
            <h3 className="text-sm font-semibold text-slate-700 mb-4">Regen – {city} (Ist-Werte, 60 Tage)</h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rainData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={9} />
                <YAxis yAxisId="mm" tick={{ fontSize: 10 }} unit=" mm" />
                <YAxis yAxisId="pct" orientation="right" tick={{ fontSize: 10 }} unit=" %" />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line yAxisId="mm" type="monotone" dataKey="Regen mm" stroke="#6366f1" dot={false} strokeWidth={2} />
                <Line yAxisId="pct" type="monotone" dataKey="Regen %" stroke="#10b981" dot={false} strokeWidth={2} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100">
              <h3 className="text-sm font-semibold text-slate-700">Rohdaten – letzte 30 Einträge</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50">
                    {["Datum", "Min °C", "Avg °C", "Max °C", "Wind m/s", "Regen mm", "Regen %", "Wetter"].map(h => (
                      <th key={h} className="text-left px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.slice(-30).reverse().map((d, i) => (
                    <tr key={d.id} className={i % 2 === 0 ? "" : "bg-slate-50/50"}>
                      <td className="px-4 py-2 text-slate-700 whitespace-nowrap">{d.target_date}</td>
                      <td className="px-4 py-2 text-slate-600">{d.temp_min_c?.toFixed(1) ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-600">{d.temp_avg_c?.toFixed(1) ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-600">{d.temp_max_c?.toFixed(1) ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-600">{d.wind_mps?.toFixed(1) ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-600">{d.rain_mm?.toFixed(1) ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-600">{d.rain_probability_pct?.toFixed(0) ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-500">{d.weather ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
