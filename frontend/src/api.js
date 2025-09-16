// frontend/src/api.js
const API_BASE = import.meta.env.VITE_API_BASE || "/api";

async function jsonOrThrow(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} – ${text || res.statusText}`);
  }
  return res.json();
}

export async function getStatusList() {
  const res = await fetch(`${API_BASE}/api/status`.replace(/\/api\/api/, "/api"));
  return jsonOrThrow(res);
}

export async function getDummyList() {
  const res = await fetch(`${API_BASE}/api/dummy`.replace(/\/api\/api/, "/api"));
  return jsonOrThrow(res);
}

export async function createDummy(message) {
  const res = await fetch(`${API_BASE}/api/dummy`.replace(/\/api\/api/, "/api"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  return jsonOrThrow(res);
}
