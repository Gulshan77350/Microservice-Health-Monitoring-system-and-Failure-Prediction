import { RISK_COLORS } from "../config";

const fmt = (v, unit = "") => (v == null ? "—" : `${v}${unit}`);

export default function ServiceCard({ service, row, busy, onSimulate, onReset }) {
  const risk = row?.risk ?? "LOW";
  const c = RISK_COLORS[risk];
  return (
    <div
      className="panel"
      style={{
        background: c.bg,
        borderColor: c.border,
        transition: "all 0.4s",
        boxShadow: risk === "HIGH" ? `0 0 16px ${c.fg}40` : "none",
      }}
    >
      <div className="panel-title">
        <span style={{ color: service.color, fontWeight: 700, fontSize: 15 }}>{service.name}</span>
        <span
          className="pill"
          style={{
            background: `${c.fg}25`,
            color: c.fg,
            borderColor: `${c.fg}60`,
            animation: risk === "HIGH" ? "flash 0.6s ease-in-out infinite alternate" : "none",
          }}
        >
          {row ? `${risk} RISK` : "NO DATA"}
        </span>
      </div>

      {row ? (
        <div className="kv" style={{ marginBottom: 14 }}>
          {[
            ["CPU", fmt(row.cpu, "%")],
            ["Memory", fmt(row.memory, "%")],
            ["Latency", fmt(row.latency, " ms")],
            ["Error rate", fmt(row.error_rate, "%")],
            ["Requests / min", fmt(row.requests)],
            ["P(failure)", row.failure_probability == null ? "—" : `${(row.failure_probability * 100).toFixed(1)}%`],
          ].map(([label, value]) => (
            <div key={label}>
              <div className="kv-label">{label}</div>
              <div className="kv-value">{value}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty" style={{ marginBottom: 14 }}>
          Waiting for the collector…
        </div>
      )}

      {row && (
        <div style={{ color: "var(--muted)", fontSize: 11, marginBottom: 12 }}>
          {row.alert ? "🔔 Alerting" : "No alert"} · heuristic cause:{" "}
          <span style={{ color: c.fg, fontWeight: 600 }}>{row.root_cause}</span>
        </div>
      )}

      <div className="row">
        <button className="btn btn-danger" style={{ flex: 1 }} disabled={busy} onClick={() => onSimulate(service)}>
          Simulate failure
        </button>
        <button className="btn btn-ok" style={{ flex: 1 }} disabled={busy} onClick={() => onReset(service)}>
          Reset
        </button>
      </div>
    </div>
  );
}
