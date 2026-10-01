import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "./api/client";
import AlertBanner from "./components/AlertBanner";
import DriftPanel from "./components/DriftPanel";
import ErrorBoundary from "./components/ErrorBoundary";
import { AnomalyFeed, EventLog } from "./components/Feeds";
import ModelPanel from "./components/ModelPanel";
import RecipientManager from "./components/RecipientManager";
import ServiceCard from "./components/ServiceCard";
import TelemetryCharts from "./components/TelemetryCharts";
import { MAX_ALERTS, MAX_LOGS, SERVICES, SLOW_POLL_MS } from "./config";
import { useLiveTelemetry } from "./hooks/useLiveTelemetry";
import { usePolling } from "./hooks/usePolling";

const nowLabel = () => new Date().toLocaleTimeString("en-GB");
const serviceName = (id) => SERVICES.find((s) => s.id === id)?.name ?? id;

export default function App() {
  const [logs, setLogs] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [importance, setImportance] = useState(null);
  const [alertConfig, setAlertConfig] = useState(null);
  const [recipientError, setRecipientError] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [events, setEvents] = useState([]);
  const [drift, setDrift] = useState(null);
  const [zThreshold, setZThreshold] = useState(null);
  const [busy, setBusy] = useState({});
  const nextId = useRef(0);

  const addLog = useCallback((msg, level = "ok") => {
    const entry = { id: nextId.current++, time: nowLabel(), msg, level };
    setLogs((prev) => [...prev.slice(-(MAX_LOGS - 1)), entry]);
  }, []);

  const onRiskChange = useCallback(
    (id, before, row) => {
      const rising = row.risk !== "LOW" && (before === "LOW" || row.risk === "HIGH");
      const pct = (row.failure_probability * 100).toFixed(1);
      addLog(`${serviceName(id)}: ${before} → ${row.risk} (p=${pct}%)`, rising ? (row.risk === "HIGH" ? "err" : "warn") : "ok");
      if (rising) {
        const alert = {
          key: `${id}-${row.timestamp}`,
          time: nowLabel(),
          service: serviceName(id),
          risk: row.risk,
          rootCause: row.root_cause,
          probability: row.failure_probability,
          horizon: row.horizon_steps,
        };
        setAlerts((prev) => [...prev, alert].slice(-MAX_ALERTS));
      }
    },
    [addLog],
  );

  const { history, latest, error: liveError } = useLiveTelemetry(onRiskChange);

  // Model metadata and recipients: once on mount.
  useEffect(() => {
    api.modelInfo().then(setModelInfo, (e) => setModelError(errorMessage(e)));
    api.featureImportance().then(setImportance, () => {});
    api.alertConfig().then(setAlertConfig, (e) => setRecipientError(errorMessage(e)));
    api.collectorHealth().then((h) => setZThreshold(h.z_threshold), () => {});
  }, []);

  // Slower feeds: anomalies, drift, collector events.
  usePolling(() => {
    api.anomalies(20).then((r) => setAnomalies(r.anomalies), () => {});
    api.drift().then(setDrift, () => {});
    api.events(20).then((r) => setEvents(r.events), () => {});
  }, SLOW_POLL_MS);

  const runServiceAction = async (service, action, label, level) => {
    setBusy((b) => ({ ...b, [service.id]: true }));
    try {
      await action(service.id);
      addLog(`${label} → ${service.name}`, level);
    } catch (e) {
      addLog(`${label} failed for ${service.name}: ${errorMessage(e)}`, "err");
    } finally {
      setBusy((b) => ({ ...b, [service.id]: false }));
    }
  };

  const addRecipient = async (email) => {
    try {
      await api.addRecipient(email);
      setAlertConfig(await api.alertConfig());
      setRecipientError(null);
      addLog("Alert recipient added");
      return true;
    } catch (e) {
      setRecipientError(errorMessage(e));
      return false;
    }
  };

  const removeRecipient = async (id) => {
    try {
      await api.removeRecipient(id);
      setAlertConfig(await api.alertConfig());
      setRecipientError(null);
      addLog("Alert recipient removed");
    } catch (e) {
      setRecipientError(errorMessage(e));
    }
  };

  const liveColor = liveError ? "#ef4444" : "#22c55e";

  return (
    <>
      <AlertBanner alerts={alerts} onDismiss={() => setAlerts([])} />
      <div className={`page${alerts.length ? " banner-offset" : ""}`}>
        <div className="header">
          <div>
            <h1>Microservice Health Monitor</h1>
            <p className="subtitle">
              Services → collector → calibrated failure-risk model → dashboard
              {modelInfo &&
                ` · model ${modelInfo.model_version} · P(failure within ${modelInfo.target.horizon_steps} collection intervals)`}
            </p>
          </div>
          <span className="pill" style={{ color: liveColor, borderColor: "#1e293b", display: "flex", gap: 6, alignItems: "center" }}>
            <span className="status-dot" style={{ background: liveColor, animation: "flash 1.4s ease-in-out infinite alternate" }} />
            {liveError ? `Collector unreachable: ${liveError}` : "Live"}
          </span>
        </div>

        <div className="grid-3">
          {SERVICES.map((s) => (
            <ServiceCard
              key={s.id}
              service={s}
              row={latest[s.id]}
              busy={!!busy[s.id]}
              onSimulate={(svc) => runServiceAction(svc, api.simulateFailure, "Failure injected", "warn")}
              onReset={(svc) => runServiceAction(svc, api.resetService, "Reset to healthy", "ok")}
            />
          ))}
        </div>

        <ErrorBoundary name="Charts">
          <TelemetryCharts history={history} latest={latest} thresholds={modelInfo?.thresholds} />
        </ErrorBoundary>

        <ErrorBoundary name="Drift">
          <DriftPanel drift={drift} events={events} />
        </ErrorBoundary>

        <div className="grid-2">
          <ErrorBoundary name="Model">
            <ModelPanel info={modelInfo} importance={importance} error={modelError} />
          </ErrorBoundary>
          <AnomalyFeed anomalies={anomalies} zThreshold={zThreshold} />
        </div>

        <div className="grid-2">
          <EventLog logs={logs} onClear={() => setLogs([])} />
          <RecipientManager config={alertConfig} error={recipientError} onAdd={addRecipient} onRemove={removeRecipient} />
        </div>
      </div>
    </>
  );
}
