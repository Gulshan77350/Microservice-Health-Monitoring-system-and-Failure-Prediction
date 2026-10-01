import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const f = (v, d = 3) => (v == null ? "—" : Number(v).toFixed(d));

function comparisonRows(info) {
  const t = info.test_metrics;
  const b = info.baselines;
  return [
    { name: `XGBoost (served, ${info.model_version})`, served: true, ...t, op: t.at_cost_optimal },
    { name: "Logistic regression", ...b.logistic_regression, op: b.logistic_regression.at_cost_optimal },
    { name: "Random forest", ...b.random_forest, op: b.random_forest.at_cost_optimal },
    { name: "Rule: error_rate > 10", ...b.rule, brier: null, op: b.rule.at_cost_optimal },
    { name: "Always alert", pr_auc: null, roc_auc: null, brier: null, op: info.trivial.always_alert },
  ];
}

export default function ModelPanel({ info, importance, error }) {
  if (error) {
    return (
      <div className="panel">
        <div className="panel-title">🤖 Model</div>
        <div className="error-text">{error}</div>
      </div>
    );
  }
  if (!info) {
    return (
      <div className="panel">
        <div className="empty">Loading model info…</div>
      </div>
    );
  }
  const t = info.test_metrics;
  const top = (importance?.feature_importance ?? []).slice(0, 8);

  return (
    <div className="panel">
      <div className="panel-title">
        <span>🤖 Model {info.model_version}</span>
        <span className="tag">{info.calibration} calibration · held-out test split</span>
      </div>

      <div className="kv" style={{ marginBottom: 14 }}>
        {[
          ["Target", `failure within ${info.target.horizon_steps} steps`],
          ["Alert threshold (cost-optimal)", f(info.thresholds.cost_optimal, 3)],
          ["HIGH threshold (max-F1)", f(info.thresholds.max_f1, 3)],
          ["Early-warning recall", f(t.early_warning.early_warning_recall, 3)],
        ].map(([label, value]) => (
          <div key={label}>
            <div className="kv-label">{label}</div>
            <div className="kv-value" style={{ fontSize: 13 }}>{value}</div>
          </div>
        ))}
      </div>

      <table className="metrics">
        <thead>
          <tr>
            <th>Model</th>
            <th>PR-AUC</th>
            <th>ROC-AUC</th>
            <th>Brier</th>
            <th>Prec</th>
            <th>Recall</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {comparisonRows(info).map((r) => (
            <tr key={r.name} className={r.served ? "served" : ""}>
              <td>{r.name}</td>
              <td>{f(r.pr_auc)}</td>
              <td>{f(r.roc_auc)}</td>
              <td>{f(r.brier)}</td>
              <td>{f(r.op.precision, 2)}</td>
              <td>{f(r.op.recall, 2)}</td>
              <td>{r.op.cost}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="note" style={{ marginTop: -6, marginBottom: 12 }}>
        Precision/recall/cost at each model&apos;s own cost-optimal threshold (FN = 10 × FP), tuned on validation only.
        At max-F1 the served model reaches P={f(t.at_max_f1.precision, 2)}, R={f(t.at_max_f1.recall, 2)}.
      </div>

      {top.length > 0 && (
        <>
          <div className="kv-label" style={{ marginBottom: 6 }}>
            Feature importance ({importance.method === "mean_abs_shap" ? "mean |SHAP|" : "gain"}, top {top.length})
          </div>
          <ResponsiveContainer width="100%" height={top.length * 20 + 20}>
            <BarChart data={top} layout="vertical" margin={{ left: 0, right: 12, top: 0, bottom: 0 }}>
              <XAxis type="number" unit="%" tick={{ fill: "#475569", fontSize: 10 }} />
              <YAxis dataKey="feature" type="category" interval={0} tick={{ fill: "#94a3b8", fontSize: 10 }} width={130} />
              <Tooltip formatter={(v) => `${v.toFixed(1)}%`} contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 12 }} />
              <Bar dataKey="importance_pct" fill="#38bdf8" radius={[0, 3, 3, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}
