import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SERVICES } from "../config";
import { mergeByTimestamp } from "../lib/series";

const TABS = ["failure probability", "cpu & memory", "latency", "error rate"];
const tooltipStyle = { background: "#1e293b", border: "1px solid #334155", fontSize: 12 };
const axisTick = { fill: "#475569", fontSize: 10 };

function SeriesChart({ data, suffix, domain, height = 220, refs = [] }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="time" tick={axisTick} interval="preserveStartEnd" />
        <YAxis domain={domain} tick={axisTick} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {refs.map((r) => (
          <ReferenceLine key={r.label} y={r.y} stroke={r.color} strokeDasharray="4 4" label={{ value: r.label, fill: r.color, fontSize: 10, position: "insideTopRight" }} />
        ))}
        {SERVICES.map((s) => (
          <Line
            key={s.id}
            type="monotone"
            dataKey={`${s.id}_${suffix}`}
            name={s.name}
            stroke={s.color}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function TelemetryCharts({ history, latest, thresholds }) {
  const [tab, setTab] = useState(TABS[0]);
  const data = useMemo(() => mergeByTimestamp(history), [history]);

  const probRefs = thresholds
    ? [
        { y: +(thresholds.cost_optimal * 100).toFixed(1), label: "alert (cost-optimal)", color: "#f59e0b" },
        { y: +(thresholds.max_f1 * 100).toFixed(1), label: "HIGH (max-F1)", color: "#ef4444" },
      ]
    : [];

  return (
    <div className="panel mb">
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={`tab${tab === t ? " active" : ""}`} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {tab === "failure probability" && (
        <>
          <div className="kv-label" style={{ marginBottom: 8 }}>
            Calibrated P(failure within the model horizon), %, as stored by the collector
          </div>
          <SeriesChart data={data} suffix="prob" domain={[0, 100]} refs={probRefs} />
        </>
      )}

      {tab === "cpu & memory" && (
        <>
          <div className="kv-label" style={{ marginBottom: 8 }}>CPU % over time</div>
          <SeriesChart data={data} suffix="cpu" domain={[0, 100]} height={180} />
          <div className="kv-label" style={{ margin: "16px 0 8px" }}>Latest CPU / memory snapshot</div>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={SERVICES.map((s) => ({ name: s.name, cpu: latest[s.id]?.cpu ?? 0, memory: latest[s.id]?.memory ?? 0 }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={axisTick} />
              <YAxis domain={[0, 100]} tick={axisTick} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="cpu" name="CPU %" fill="#38bdf8" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              <Bar dataKey="memory" name="Memory %" fill="#a78bfa" radius={[3, 3, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}

      {tab === "latency" && (
        <>
          <div className="kv-label" style={{ marginBottom: 8 }}>Mean /process latency (ms), measured in each service</div>
          <SeriesChart data={data} suffix="lat" />
        </>
      )}

      {tab === "error rate" && (
        <>
          <div className="kv-label" style={{ marginBottom: 8 }}>% of recent /process requests returning 5xx</div>
          <SeriesChart data={data} suffix="err" domain={[0, 100]} />
        </>
      )}
    </div>
  );
}
