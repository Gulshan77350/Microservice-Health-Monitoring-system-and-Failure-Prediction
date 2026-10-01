import { useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api/client";
import { MAX_POINTS, POLL_MS, SERVICES } from "../config";
import { usePolling } from "./usePolling";

export function toPoint(row) {
  return {
    timestamp: row.timestamp,
    time: new Date(row.timestamp).toLocaleTimeString("en-GB"),
    cpu: row.cpu,
    memory: row.memory,
    latency: row.latency,
    error_rate: row.error_rate,
    requests: row.requests,
    probability: row.failure_probability == null ? null : +(row.failure_probability * 100).toFixed(1),
    risk: row.risk ?? null,
    alert: row.alert ?? null,
  };
}

export function appendPoints(prev, predictions) {
  let changed = false;
  const next = { ...prev };
  for (const { id } of SERVICES) {
    const row = predictions[id];
    if (!row) continue;
    const series = prev[id] ?? [];
    if (series.at(-1)?.timestamp === row.timestamp) continue;
    next[id] = [...series.slice(-(MAX_POINTS - 1)), toPoint(row)];
    changed = true;
  }
  return changed ? next : prev;
}

/**
 * Live + historical telemetry, read entirely from the collector (which is the
 * only caller of the prediction API). `onRiskChange(service, before, row)` is
 * called once per risk transition.
 */
export function useLiveTelemetry(onRiskChange) {
  const [history, setHistory] = useState({});
  const [latest, setLatest] = useState({});
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const prevRisk = useRef({});

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      SERVICES.map(({ id }) =>
        api
          .history(id, MAX_POINTS)
          .then((r) => [id, r.data.map(toPoint)])
          .catch(() => [id, []]),
      ),
    ).then((entries) => {
      if (cancelled) return;
      setHistory(Object.fromEntries(entries));
      setReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  usePolling(
    async () => {
      try {
        const { predictions } = await api.latestPredictions();
        setError(null);
        setLatest(predictions);
        setHistory((prev) => appendPoints(prev, predictions));
        for (const { id } of SERVICES) {
          const row = predictions[id];
          if (!row) continue;
          const before = prevRisk.current[id];
          if (before !== undefined && before !== row.risk) onRiskChange(id, before, row);
          prevRisk.current[id] = row.risk;
        }
      } catch (err) {
        setError(errorMessage(err));
      }
    },
    POLL_MS,
    ready,
  );

  return { history, latest, error };
}
