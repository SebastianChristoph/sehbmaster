import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { Shield, LogOut, Lock, User, RefreshCw } from "lucide-react";
import { api } from "../api/client";

const projects = [
  { name: "Bildwatch", schema: "bild" },
];

export function Admin() {
  const { isAdmin, login, logout } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [schemas, setSchemas] = useState<string[] | null>(null);

  useEffect(() => {
    if (isAdmin) {
      api.getInfo().then(d => setSchemas(d.schemas)).catch(() => setSchemas([]));
    }
  }, [isAdmin]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(username, password);
    } catch {
      setError("Ungültige Anmeldedaten");
    } finally {
      setLoading(false);
    }
  };

  if (!isAdmin) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="bg-white border border-slate-200 rounded-xl p-8 w-full max-w-sm shadow-sm">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-indigo-50 rounded-lg">
              <Shield size={20} className="text-indigo-500" />
            </div>
            <div>
              <h1 className="text-base font-semibold text-slate-800">Admin-Bereich</h1>
              <p className="text-xs text-slate-500">Anmeldung erforderlich</p>
            </div>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Benutzername</label>
              <div className="relative">
                <User size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-400"
                  placeholder="admin"
                  autoComplete="username"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Passwort</label>
              <div className="relative">
                <Lock size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-400"
                  placeholder="••••••••"
                  autoComplete="current-password"
                />
              </div>
            </div>
            {error && <p className="text-xs text-rose-500">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-60"
            >
              {loading ? "..." : "Anmelden"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Admin</h1>
          <p className="text-sm text-slate-500 mt-0.5">Administrationsbereich</p>
        </div>
        <button
          onClick={logout}
          className="flex items-center gap-2 text-sm text-slate-500 hover:text-rose-600 border border-slate-200 px-3 py-1.5 rounded-lg hover:border-rose-200 transition-colors"
        >
          <LogOut size={14} />
          Abmelden
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-lg p-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">System</h2>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-slate-500">Backend</dt>
              <dd className="text-slate-800 font-medium">FastAPI @ :8000</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">Datenbank</dt>
              <dd className="text-slate-800 font-medium">PostgreSQL 16</dd>
            </div>
            <div className="flex justify-between items-start">
              <dt className="text-slate-500">DB-Schemas</dt>
              <dd className="text-slate-800 font-medium text-right">
                {schemas === null ? (
                  <RefreshCw size={12} className="animate-spin inline" />
                ) : schemas.length === 0 ? (
                  <span className="text-slate-400">—</span>
                ) : (
                  <span className="font-mono text-xs">{schemas.join(", ")}</span>
                )}
              </dd>
            </div>
          </dl>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">Projekte</h2>
          <div className="space-y-2 text-sm">
            {projects.map(p => (
              <div key={p.name} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
                <span className="text-slate-700 font-medium">{p.name}</span>
                <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded font-mono">{p.schema}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
        <p className="text-sm text-amber-700">
          <span className="font-semibold">Hinweis:</span> Bitte ändere das Standard-Passwort in der <code className="font-mono bg-amber-100 px-1 rounded">.env</code>-Datei auf dem Server.
        </p>
      </div>
    </div>
  );
}
