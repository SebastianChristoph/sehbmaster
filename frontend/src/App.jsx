import { useEffect, useState } from "react";
import "./App.css";
import { getStatusList, getDummyList, createDummy } from "./api";

function App() {
  const [loading, setLoading] = useState(true);
  const [statusRows, setStatusRows] = useState([]);
  const [dummyRows, setDummyRows] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [error, setError] = useState("");

  async function loadAll() {
    setLoading(true);
    setError("");
    try {
      const [statusData, dummyData] = await Promise.all([
        getStatusList(),
        getDummyList(),
      ]);
      // optionale Sortierung
      setStatusRows(
        [...statusData].sort((a, b) => a.raspberry.localeCompare(b.raspberry))
      );
      setDummyRows([...dummyData].reverse()); // neuestes nach oben, falls du später Timestamps hast
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
  }, []);

  async function handleAddDummy(e) {
    e.preventDefault();
    if (!newMessage.trim()) return;
    try {
      await createDummy(newMessage.trim());
      setNewMessage("");
      await loadAll();
    } catch (e) {
      setError(e.message || String(e));
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>sehbmaster – Übersicht</h1>
        <div className="actions">
          <button onClick={loadAll} disabled={loading}>
            {loading ? "Lade…" : "Neu laden"}
          </button>
        </div>
      </header>

      {error && <div className="error">⚠️ {error}</div>}

      <section className="card">
        <h2>Status (Schema: <code>status.status</code>)</h2>
        <p className="muted">
          Übersicht deiner Raspberrys (aus dem Backend gelesen).
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Raspberry</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {statusRows.length === 0 ? (
                <tr>
                  <td colSpan="2" className="muted">
                    Keine Einträge vorhanden.
                  </td>
                </tr>
              ) : (
                statusRows.map((row) => (
                  <tr key={row.raspberry}>
                    <td>{row.raspberry}</td>
                    <td>{row.status}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Dummy Messages (Schema: <code>dummy.dummy_table</code>)</h2>
        <form onSubmit={handleAddDummy} className="form">
          <input
            type="text"
            placeholder="Neue Nachricht…"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
          />
          <button type="submit" disabled={!newMessage.trim()}>
            Hinzufügen
          </button>
        </form>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Message</th>
              </tr>
            </thead>
            <tbody>
              {dummyRows.length === 0 ? (
                <tr>
                  <td className="muted">Keine Einträge vorhanden.</td>
                </tr>
              ) : (
                dummyRows.map((row, i) => (
                  <tr key={`${row.message}-${i}`}>
                    <td>{row.message}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="footer">
        <span className="muted">
          API: {import.meta.env.VITE_API_BASE || "http://localhost:8000"}
        </span>
      </footer>
    </div>
  );
}

export default App;
