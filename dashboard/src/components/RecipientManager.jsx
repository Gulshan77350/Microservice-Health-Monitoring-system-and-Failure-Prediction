import { useState } from "react";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function RecipientManager({ config, error, onAdd, onRemove }) {
  const [input, setInput] = useState("");
  const [localError, setLocalError] = useState(null);

  const submit = async () => {
    const email = input.trim();
    if (!EMAIL_RE.test(email)) {
      setLocalError("Enter a valid email address");
      return;
    }
    setLocalError(null);
    if (await onAdd(email)) setInput("");
  };

  return (
    <div className="panel">
      <div className="panel-title">
        <span>📧 Alert recipients</span>
        {config && (
          <span className="tag" style={{ color: config.alerts_enabled ? "#22c55e" : "#f59e0b" }}>
            {config.alerts_enabled ? "email alerts enabled" : "email alerts disabled (SMTP not configured)"}
          </span>
        )}
      </div>
      <div className="chips">
        {(config?.recipients ?? []).map((r) => (
          <span key={r.id} className="chip">
            {r.email}
            <button className="btn-ghost" style={{ cursor: "pointer", fontSize: 14 }} aria-label={`Remove ${r.email}`} onClick={() => onRemove(r.id)}>
              ×
            </button>
          </span>
        ))}
        {config && !config.recipients.length && <span className="empty">No recipients</span>}
      </div>
      <div className="row">
        <input
          className="input"
          type="email"
          placeholder="Add email…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        <button className="btn btn-primary" onClick={submit}>
          Add
        </button>
      </div>
      {(localError || error) && <div className="error-text" style={{ marginTop: 8 }}>{localError || error}</div>}
      <div className="note">
        Addresses are stored by the prediction API and shown masked. An email is sent when a service enters the
        alerting state (cost-optimal threshold), at most once per cooldown window.
      </div>
    </div>
  );
}
