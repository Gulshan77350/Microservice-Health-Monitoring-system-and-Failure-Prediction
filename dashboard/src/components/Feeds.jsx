import { useEffect, useRef } from "react";

const LEVEL_COLORS = { err: "#ef4444", warn: "#f59e0b", ok: "#22c55e" };

export function AnomalyFeed({ anomalies, zThreshold }) {
  return (
    <div className="panel">
      <div className="panel-title">
        <span>
          🔍 Anomaly detector <span className="tag">|z| &gt; {zThreshold ?? "?"}σ vs previous 20 points</span>
        </span>
      </div>
      <div className="feed">
        {!anomalies.length && <div className="empty">No anomalies detected yet</div>}
        {anomalies.map((a) => (
          <div key={`${a.timestamp}-${a.service}-${a.metric}`} className="feed-item" style={{ background: "#1a1200" }}>
            <span style={{ color: "#f59e0b", fontWeight: 600 }}>[{a.service}]</span>
            <span style={{ color: "var(--muted)" }}>{a.metric}</span>
            <span>{a.value}</span>
            <span style={{ color: "var(--faint)" }}>
              (mean={a.mean}, z={a.z_score})
            </span>
            <span style={{ color: "var(--faint)", marginLeft: "auto" }}>{new Date(a.timestamp).toLocaleTimeString("en-GB")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function EventLog({ logs, onClear }) {
  const boxRef = useRef(null);
  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [logs]);

  return (
    <div className="panel">
      <div className="panel-title">
        <span>📋 Event log</span>
        <button className="btn btn-ghost" onClick={onClear}>
          Clear
        </button>
      </div>
      <div className="feed" ref={boxRef}>
        {!logs.length && <span className="empty">No events yet</span>}
        {logs.map((l) => (
          <div key={l.id} className="feed-item">
            <span className="status-dot" style={{ background: LEVEL_COLORS[l.level], flexShrink: 0 }} />
            <span style={{ color: "var(--faint)", whiteSpace: "nowrap" }}>{l.time}</span>
            <span style={{ color: "var(--muted)" }}>{l.msg}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
