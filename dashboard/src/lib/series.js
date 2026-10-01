import { SERVICES } from "../config";

/** Merge per-service series by collector timestamp (all services share one timestamp per cycle). */
export function mergeByTimestamp(history) {
  const rows = new Map();
  for (const { id } of SERVICES) {
    for (const p of history[id] ?? []) {
      const row = rows.get(p.timestamp) ?? { timestamp: p.timestamp, time: p.time };
      row[`${id}_prob`] = p.probability;
      row[`${id}_cpu`] = p.cpu;
      row[`${id}_mem`] = p.memory;
      row[`${id}_lat`] = p.latency;
      row[`${id}_err`] = p.error_rate;
      rows.set(p.timestamp, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}
