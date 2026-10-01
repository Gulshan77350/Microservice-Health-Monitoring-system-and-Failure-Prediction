const LEVELS = {
  stable: { color: "#22c55e", label: "Stable" },
  moderate_shift: { color: "#f59e0b", label: "Moderate shift" },
  significant_shift: { color: "#ef4444", label: "Significant shift" },
  insufficient_data: { color: "#64748b", label: "Warming up" },
};

export default function DriftPanel({ drift, events }) {
  if (!drift) return null;
  const overall = LEVELS[drift.overall_status] ?? LEVELS.insufficient_data;
  const retrain = events.find((e) => e.kind === "retrain_recommended");

  return (
    <div className="panel mb">
      <div className="panel-title">
        <span>
          📉 Input drift (PSI vs. training data) <span className="tag">min {drift.min_live_samples} samples</span>
        </span>
        <span className="pill" style={{ color: overall.color, borderColor: `${overall.color}60`, background: `${overall.color}15` }}>
          {overall.label}
          {drift.max_psi != null && ` · max PSI ${drift.max_psi}`}
        </span>
      </div>

      <div className="drift-grid">
        {Object.entries(drift.per_feature ?? {}).map(([feature, info]) => {
          const lvl = LEVELS[info.level] ?? LEVELS.insufficient_data;
          return (
            <div key={feature} className="drift-cell" style={{ borderColor: `${lvl.color}40` }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 600, textTransform: "uppercase" }}>{feature}</span>
                <span style={{ color: lvl.color, fontSize: 10, fontWeight: 700 }}>{lvl.label}</span>
              </div>
              <div style={{ color: "var(--muted)", fontSize: 11 }}>
                PSI: <strong style={{ color: lvl.color }}>{info.psi ?? "—"}</strong>
              </div>
              <div style={{ color: "var(--faint)", fontSize: 10, marginTop: 2 }}>{info.live_samples} live samples</div>
            </div>
          );
        })}
      </div>

      {retrain && (
        <div className="note" style={{ color: "#fca5a5" }}>
          [{new Date(retrain.timestamp).toLocaleString("en-GB")}] {retrain.message}
        </div>
      )}
      <div className="note">
        PSI &lt; 0.1 stable · 0.1–0.25 investigate · &gt; 0.25 retrain recommended. Live telemetry from these demo
        services is expected to drift: the model was trained on synthetic data with a different load profile.
      </div>
    </div>
  );
}
