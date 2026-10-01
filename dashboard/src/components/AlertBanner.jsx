export default function AlertBanner({ alerts, onDismiss }) {
  if (!alerts.length) return null;
  return (
    <div className="banner" role="alert">
      <span style={{ fontSize: 20, animation: "flash 0.8s ease-in-out infinite alternate" }}>⚠️</span>
      <div style={{ flex: 1 }}>
        {alerts.map((a) => (
          <div key={a.key} className="banner-line">
            [{a.time}] {a.service}: {a.risk} risk — {a.rootCause} (p={(a.probability * 100).toFixed(1)}% within{" "}
            {a.horizon} intervals)
          </div>
        ))}
      </div>
      <button className="btn btn-danger" onClick={onDismiss}>
        Dismiss
      </button>
    </div>
  );
}
