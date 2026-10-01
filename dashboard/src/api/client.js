import axios from "axios";

// All URLs are same-origin. In production Nginx routes them (and injects the
// X-API-Key for the few mutating routes it exposes); in development the Vite
// proxy does the same. The browser bundle never contains the key.
const http = axios.create({ timeout: 5000 });
const body = (promise) => promise.then((r) => r.data);

export const api = {
  // Collector (read-only): live + historical metrics and the predictions it stored
  latestPredictions: () => body(http.get("/collector/predictions/latest")),
  history: (service, limit) => body(http.get("/collector/history", { params: { service, limit } })),
  anomalies: (limit = 20) => body(http.get("/collector/anomalies", { params: { limit } })),
  events: (limit = 20) => body(http.get("/collector/events", { params: { limit } })),
  collectorHealth: () => body(http.get("/collector/health")),

  // Prediction API: model metadata, drift, alert recipients
  modelInfo: () => body(http.get("/api/model-info")),
  featureImportance: () => body(http.get("/api/feature-importance")),
  drift: () => body(http.get("/api/drift")),
  alertConfig: () => body(http.get("/api/alert-config")),
  addRecipient: (email) => body(http.post("/api/alert-config/recipients", { email })),
  removeRecipient: (id) => body(http.delete(`/api/alert-config/recipients/${id}`)),

  // Simulated services: failure injection
  simulateFailure: (service) => body(http.post(`/api/${service}/simulate_failure`)),
  resetService: (service) => body(http.post(`/api/${service}/reset`)),
};

export function errorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
  return detail ?? err?.message ?? "Request failed";
}
