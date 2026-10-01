import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, RadarChart,
  PolarGrid, PolarAngleAxis, Radar,
} from "recharts";

// ─── Config ────────────────────────────────────────────────────────────────
const SERVICES = [
  { name: "Payment",      metricsUrl: "/api/payment/metrics",     color: "#38bdf8" },
  { name: "Order",        metricsUrl: "/api/order/metrics",        color: "#a78bfa" },
  { name: "Notification", metricsUrl: "/api/notification/metrics", color: "#34d399" },
];
const POLL_MS     = 5000;
const MAX_HISTORY = 30;
const RISK_ORDER  = { LOW: 0, MEDIUM: 1, HIGH: 2 };

// ─── Helpers ───────────────────────────────────────────────────────────────
const riskColor  = r => r === "HIGH" ? "#ef4444" : r === "MEDIUM" ? "#f59e0b" : "#22c55e";
const riskBg     = r => r === "HIGH" ? "#1f0a0a" : r === "MEDIUM" ? "#1a1200" : "#071a0e";
const riskBorder = r => r === "HIGH" ? "#7f1d1d" : r === "MEDIUM" ? "#78350f" : "#14532d";
const nowLabel   = () => new Date().toLocaleTimeString("en-GB");

function isAboveThreshold(risk) { return RISK_ORDER[risk] >= RISK_ORDER["MEDIUM"]; }

// ─── Components ────────────────────────────────────────────────────────────

function AlertBanner({ alerts, onDismiss }) {
  if (!alerts.length) return null;
  return (
    <div style={{
      position:"fixed", top:0, left:0, right:0, zIndex:999,
      background:"#7f1d1d", borderBottom:"2px solid #ef4444",
      padding:"10px 20px", display:"flex", alignItems:"center", gap:12,
      animation:"flashBanner 0.8s ease-in-out infinite alternate"
    }}>
      <span style={{fontSize:20}}>⚠️</span>
      <div style={{flex:1}}>
        {alerts.map((a,i) => (
          <div key={i} style={{color:"#fecaca",fontSize:13,fontWeight:600}}>
            [{a.time}] {a.service}: {a.risk} RISK — {a.root_cause} (prob: {(a.probability*100).toFixed(1)}%)
          </div>
        ))}
      </div>
      <button onClick={onDismiss} style={{
        background:"none", border:"1px solid #ef4444", color:"#fca5a5",
        padding:"4px 12px", borderRadius:6, cursor:"pointer", fontSize:12
      }}>Dismiss</button>
    </div>
  );
}

function ServiceCard({ service, latest, prediction, onSimulate, onReset }) {
  const risk = prediction?.risk ?? "LOW";
  return (
    <div style={{
      background:riskBg(risk), border:`1px solid ${riskBorder(risk)}`,
      borderRadius:12, padding:"18px 20px", transition:"all 0.4s",
      ...(risk==="HIGH" ? {boxShadow:`0 0 16px ${riskColor(risk)}40`} : {})
    }}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
        <span style={{color:service.color,fontWeight:700,fontSize:15}}>{service.name}</span>
        <span style={{
          background:riskColor(risk)+"25", color:riskColor(risk),
          border:`1px solid ${riskColor(risk)}60`,
          padding:"2px 10px", borderRadius:20, fontSize:11, fontWeight:700,
          animation: risk==="HIGH" ? "flashPill 0.6s ease-in-out infinite alternate" : "none"
        }}>{risk} RISK</span>
      </div>
      {latest ? (
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"8px 16px",marginBottom:14}}>
          {[
            ["CPU",        latest.cpu+"%" ],
            ["Memory",     latest.memory+"%"],
            ["Latency",    latest.latency+"ms"],
            ["Error Rate", latest.error_rate+"%"],
            ["Requests",   latest.requests],
            ["Prob",       prediction ? (prediction.failure_probability*100).toFixed(1)+"%" : "—"],
          ].map(([label,val]) => (
            <div key={label}>
              <div style={{color:"#64748b",fontSize:10,marginBottom:1}}>{label}</div>
              <div style={{color:"#e2e8f0",fontSize:14,fontWeight:600}}>{val}</div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{color:"#475569",fontSize:13,marginBottom:14}}>Connecting…</div>
      )}
      {prediction && (
        <div style={{color:"#94a3b8",fontSize:11,marginBottom:12}}>
          Root cause: <span style={{color:riskColor(risk),fontWeight:600}}>{prediction.root_cause}</span>
        </div>
      )}
      <div style={{display:"flex",gap:8}}>
        <button onClick={() => onSimulate(service)} style={{
          flex:1, background:"#7f1d1d", border:"1px solid #991b1b",
          color:"#fca5a5", borderRadius:6, padding:"5px 0", fontSize:11, cursor:"pointer", fontWeight:600
        }}>Simulate Failure</button>
        <button onClick={() => onReset(service)} style={{
          flex:1, background:"#14532d", border:"1px solid #166534",
          color:"#86efac", borderRadius:6, padding:"5px 0", fontSize:11, cursor:"pointer", fontWeight:600
        }}>Reset</button>
      </div>
    </div>
  );
}

function ModelInfoPanel({ modelInfo, featureImportance }) {
  if (!modelInfo) return (
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:"18px 20px"}}>
      <div style={{color:"#475569",fontSize:13}}>Loading model info…</div>
    </div>
  );
  return (
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:"18px 20px"}}>
      <div style={{color:"#94a3b8",fontSize:13,fontWeight:600,marginBottom:14}}>🤖 Model Info</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"8px 20px",marginBottom:16}}>
        {[
          ["Algorithm",  modelInfo.algorithm],
          ["Accuracy",   modelInfo.accuracy != null ? (modelInfo.accuracy*100).toFixed(1)+"%" : "—"],
          ["F1 Score",   modelInfo.f1_score != null ? modelInfo.f1_score.toFixed(4) : "—"],
          ["Precision",  modelInfo.precision != null ? modelInfo.precision.toFixed(4) : "—"],
          ["Recall",     modelInfo.recall != null ? modelInfo.recall.toFixed(4) : "—"],
          ["CV F1 Mean", modelInfo.cv_f1_mean != null ? modelInfo.cv_f1_mean.toFixed(4) : "—"],
          ["Train Samples", modelInfo.train_samples],
          ["Test Samples",  modelInfo.test_samples],
        ].map(([label,val]) => (
          <div key={label}>
            <div style={{color:"#475569",fontSize:10,marginBottom:1}}>{label}</div>
            <div style={{color:"#e2e8f0",fontSize:13,fontWeight:600}}>{val ?? "—"}</div>
          </div>
        ))}
      </div>
      {featureImportance?.length > 0 && (
        <>
          <div style={{color:"#64748b",fontSize:11,marginBottom:8}}>Feature Importance</div>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart
              data={featureImportance}
              layout="vertical"
              margin={{left:0,right:12,top:0,bottom:0}}
            >
              <XAxis type="number" domain={[0,100]} tick={{fill:"#475569",fontSize:10}} unit="%" />
              <YAxis dataKey="feature" type="category" tick={{fill:"#94a3b8",fontSize:11}} width={80} />
              <Tooltip
                formatter={v => v.toFixed(2)+"%"}
                contentStyle={{background:"#1e293b",border:"1px solid #334155",fontSize:12}}
              />
              <Bar dataKey="importance_pct" fill="#38bdf8" radius={[0,3,3,0]} name="Importance %" />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

function AnomalyFeed({ anomalies }) {
  return (
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:"18px 20px"}}>
      <div style={{color:"#94a3b8",fontSize:13,fontWeight:600,marginBottom:10}}>
        🔍 Anomaly Detector
        <span style={{
          marginLeft:8, fontSize:10, background:"#1e293b",
          border:"1px solid #334155", borderRadius:10, padding:"1px 8px", color:"#64748b"
        }}>z-score &gt; 2σ</span>
      </div>
      <div style={{maxHeight:160,overflowY:"auto",display:"flex",flexDirection:"column",gap:4}}>
        {!anomalies.length && (
          <div style={{color:"#334155",fontSize:12}}>No anomalies detected yet</div>
        )}
        {anomalies.map((a,i) => (
          <div key={i} style={{
            background:"#1a1200", border:"1px solid #78350f",
            borderRadius:6, padding:"6px 10px", fontSize:12
          }}>
            <span style={{color:"#f59e0b",fontWeight:600}}>[{a.service}]</span>
            <span style={{color:"#94a3b8",marginLeft:6}}>{a.metric}</span>
            <span style={{color:"#e2e8f0",marginLeft:6}}>{a.value}</span>
            <span style={{color:"#475569",marginLeft:6}}>
              (mean={a.mean}, z={a.z_score})
            </span>
            <span style={{color:"#475569",marginLeft:8,float:"right"}}>{a.timestamp?.slice(11,19)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EventLog({ logs, onClear }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({behavior:"smooth"}); }, [logs]);
  return (
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:"18px 20px"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
        <span style={{color:"#94a3b8",fontSize:13,fontWeight:600}}>📋 Event Log</span>
        <button onClick={onClear} style={{background:"none",border:"none",color:"#475569",cursor:"pointer",fontSize:12}}>Clear</button>
      </div>
      <div style={{maxHeight:160,overflowY:"auto",display:"flex",flexDirection:"column",gap:4}}>
        {!logs.length && <span style={{color:"#334155",fontSize:12}}>No events yet</span>}
        {logs.map((l,i) => (
          <div key={i} style={{display:"flex",gap:8,fontSize:12,background:"#1e293b",borderRadius:6,padding:"4px 10px"}}>
            <span style={{
              width:8,height:8,borderRadius:"50%",marginTop:4,flexShrink:0,
              background: l.level==="err" ? "#ef4444" : l.level==="warn" ? "#f59e0b" : "#22c55e"
            }} />
            <span style={{color:"#475569",whiteSpace:"nowrap"}}>{l.time}</span>
            <span style={{color:"#94a3b8"}}>{l.msg}</span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function EmailManager({ emails, onAdd, onRemove }) {
  const [input, setInput] = useState("");
  return (
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:"18px 20px"}}>
      <div style={{color:"#94a3b8",fontSize:13,fontWeight:600,marginBottom:12}}>📧 Alert Recipients</div>
      <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:12,minHeight:32}}>
        {emails.map((e,i) => (
          <span key={i} style={{
            background:"#1e293b",border:"1px solid #334155",
            borderRadius:20,padding:"3px 12px",fontSize:12,color:"#94a3b8",
            display:"flex",alignItems:"center",gap:6
          }}>
            {e}
            <button onClick={() => onRemove(i)} style={{background:"none",border:"none",color:"#475569",cursor:"pointer",fontSize:14,padding:0}}>×</button>
          </span>
        ))}
        {!emails.length && <span style={{color:"#334155",fontSize:12}}>No recipients</span>}
      </div>
      <div style={{display:"flex",gap:8}}>
        <input
          type="email" placeholder="Add email…" value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if(e.key==="Enter"){onAdd(input);setInput("");} }}
          style={{
            flex:1,background:"#1e293b",border:"1px solid #334155",
            borderRadius:6,padding:"6px 12px",color:"#e2e8f0",fontSize:13
          }}
        />
        <button onClick={() => {onAdd(input);setInput("");}} style={{
          background:"#1e3a5f",border:"1px solid #1d4ed8",
          color:"#93c5fd",borderRadius:6,padding:"6px 14px",cursor:"pointer",fontSize:12,fontWeight:600
        }}>Add</button>
      </div>
      <div style={{color:"#475569",fontSize:11,marginTop:8}}>
        enter your email address to get alert notification
      </div>
    </div>
  );
}

function DriftPanel({ driftInfo }) {
  if (!driftInfo || driftInfo.status !== "ok") return null;
  const statusColor = s => s === "significant_shift" ? "#ef4444" : s === "moderate_shift" ? "#f59e0b" : "#22c55e";
  const statusLabel = s => s === "significant_shift" ? "Significant Shift" : s === "moderate_shift" ? "Moderate Shift" : "Stable";

  return (
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:"18px 20px",marginBottom:20}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
        <div style={{color:"#94a3b8",fontSize:13,fontWeight:600,display:"flex",alignItems:"center",gap:8}}>
          <span>📉 Model Input Drift Monitor (PSI)</span>
          <span style={{fontSize:10,background:"#1e293b",border:"1px solid #334155",borderRadius:10,padding:"1px 8px",color:"#64748b"}}>
            Population Stability Index
          </span>
        </div>
        <div style={{
          display:"flex",alignItems:"center",gap:6,
          background:statusColor(driftInfo.overall_status)+"15",
          border:`1px solid ${statusColor(driftInfo.overall_status)}40`,
          padding:"3px 10px",borderRadius:20,fontSize:11,fontWeight:700,
          color:statusColor(driftInfo.overall_status)
        }}>
          <span style={{width:6,height:6,borderRadius:"50%",background:statusColor(driftInfo.overall_status)}} />
          Overall: {statusLabel(driftInfo.overall_status)} (Max PSI: {driftInfo.max_psi})
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:10}}>
        {Object.entries(driftInfo.per_feature || {}).map(([feat, info]) => {
          const color = statusColor(info.level);
          const badgeText = info.level === "stable" ? "STABLE" : info.level === "moderate_shift" ? "WARN" : "SHIFT";
          return (
            <div key={feat} style={{
              background:"#1e293b",border:`1px solid ${color}40`,
              borderRadius:8,padding:"10px 12px"
            }}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
                <span style={{color:"#e2e8f0",fontSize:12,fontWeight:600,textTransform:"uppercase"}}>{feat}</span>
                <span style={{
                  background:color+"25",color:color,border:`1px solid ${color}60`,
                  fontSize:9,fontWeight:700,padding:"1px 6px",borderRadius:4
                }}>
                  {badgeText}
                </span>
              </div>
              <div style={{color:"#94a3b8",fontSize:11}}>
                PSI: <strong style={{color:color}}>{info.psi}</strong>
              </div>
              <div style={{color:"#475569",fontSize:10,marginTop:2}}>
                {info.live_samples_in_window} samples
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [history,         setHistory]         = useState(SERVICES.map(() => []));
  const [latest,          setLatest]          = useState(SERVICES.map(() => null));
  const [preds,           setPreds]           = useState(SERVICES.map(() => null));
  const [alerts,          setAlerts]          = useState([]);
  const [logs,            setLogs]            = useState([]);
  const [emails,          setEmails]          = useState(["alerts@example.com"]);
  const [activeTab,       setActiveTab]       = useState("overview");
  const [modelInfo,       setModelInfo]       = useState(null);
  const [featureImp,      setFeatureImp]      = useState([]);
  const [anomalies,       setAnomalies]       = useState([]);
  const [driftInfo,       setDriftInfo]       = useState(null);
  const [historyLoaded,   setHistoryLoaded]   = useState(false);

  const prevRiskRef = useRef(SERVICES.map(() => "LOW"));

  const addLog = useCallback((msg, level="ok") => {
    setLogs(prev => [...prev.slice(-99), {time: nowLabel(), msg, level}]);
  }, []);

  // ── Load model info + feature importance + alert config on mount ───────
  useEffect(() => {
    axios.get("/api/model-info").then(r => setModelInfo(r.data)).catch(()=>{});
    axios.get("/api/feature-importance").then(r => {
      setFeatureImp(r.data.feature_importance ?? []);
    }).catch(()=>{});
    axios.get("/api/alert-config").then(r => {
      if (r.data.recipients && r.data.recipients.length > 0) {
        setEmails(r.data.recipients);
      }
    }).catch(()=>{});
  }, []);

  // ── Load historical data from collector SQLite on mount ────────────────
  useEffect(() => {
    async function loadHistory() {
      const newHist = SERVICES.map(() => []);
      for (let i = 0; i < SERVICES.length; i++) {
        try {
          const svcName = SERVICES[i].name.toLowerCase();
          const { data } = await axios.get(
            `/collector/history?service=${svcName}&limit=${MAX_HISTORY}`
          );
          newHist[i] = data.data.map(row => {
            const prob = row.cpu != null ? +(Math.min(95, Math.max(3, (row.cpu * 0.15 + row.memory * 0.15 + row.error_rate * 3.5 + row.latency * 0.04))).toFixed(1)) : 5.0;
            return {
              time:        row.timestamp.slice(11,19),
              cpu:         row.cpu,
              memory:      row.memory,
              latency:     row.latency,
              error_rate:  row.error_rate,
              requests:    row.requests,
              probability: prob,
              risk:        prob > 50 ? "HIGH" : prob > 20 ? "MEDIUM" : "LOW",
            };
          });
        } catch { /* collector may not be running yet */ }
      }
      setHistory(newHist);
      setHistoryLoaded(true);
      addLog("Historical data loaded from collector DB", "ok");
    }
    loadHistory();
  }, []); // eslint-disable-line

  // ── Poll anomalies & drift every 10s ───────────────────────────────────
  useEffect(() => {
    const fetchAnomaliesAndDrift = () => {
      axios.get("/collector/anomalies?limit=20")
        .then(r => setAnomalies(r.data.anomalies ?? []))
        .catch(()=>{});
      axios.get("/api/drift")
        .then(r => setDriftInfo(r.data))
        .catch(()=>{});
    };
    fetchAnomaliesAndDrift();
    const id = setInterval(fetchAnomaliesAndDrift, 10000);
    return () => clearInterval(id);
  }, []);

  // ── Live polling — batch predict ───────────────────────────────────────
  const fetchAll = useCallback(async () => {
    const newLatest = [...latest];
    const newPreds  = [...preds];
    const newHist   = history.map(h => [...h]);
    const batchPayload = [];

    // Fetch raw metrics from all services in parallel
    await Promise.all(SERVICES.map(async (svc, i) => {
      try {
        const { data: m } = await axios.get(svc.metricsUrl);
        newLatest[i] = m;
        batchPayload.push({
          service: svc.name.toLowerCase(),
          metrics: {
            cpu: m.cpu, memory: m.memory, latency: m.latency,
            requests: m.requests, error_rate: m.error_rate,
          }
        });
      } catch {
        addLog(`Failed to reach ${svc.name}`, "err");
      }
    }));

    // Single batch predict call instead of 3 separate calls
    if (batchPayload.length > 0) {
      try {
        const { data: batchResult } = await axios.post("/api/predict/batch", batchPayload);
        const results = batchResult.results ?? {};
        const newAlertsBatch = [];

        SERVICES.forEach((svc, i) => {
          const p = results[svc.name.toLowerCase()];
          if (!p) return;
          newPreds[i] = p;

          const point = {
            time:        nowLabel(),
            cpu:         newLatest[i]?.cpu,
            memory:      newLatest[i]?.memory,
            latency:     newLatest[i]?.latency,
            error_rate:  newLatest[i]?.error_rate,
            requests:    newLatest[i]?.requests,
            probability: +(p.failure_probability * 100).toFixed(1),
            risk:        p.risk,
          };
          newHist[i] = [...newHist[i].slice(-(MAX_HISTORY-1)), point];

          if (isAboveThreshold(p.risk) && p.risk !== prevRiskRef.current[i]) {
            newAlertsBatch.push({
              time: nowLabel(), service: svc.name,
              risk: p.risk, root_cause: p.root_cause,
              probability: p.failure_probability,
            });
            addLog(`${svc.name} → ${p.risk} risk (${p.root_cause})`, p.risk==="HIGH" ? "err" : "warn");
          } else if (!isAboveThreshold(p.risk) && isAboveThreshold(prevRiskRef.current[i])) {
            addLog(`${svc.name} → recovered (${p.risk})`, "ok");
          }
          prevRiskRef.current[i] = p.risk;
        });

        if (newAlertsBatch.length) setAlerts(prev => [...prev, ...newAlertsBatch]);
      } catch {
        addLog("Batch predict failed — falling back", "warn");
      }
    }

    setLatest(newLatest);
    setPreds(newPreds);
    setHistory(newHist);
  }, [latest, preds, history, addLog]);

  useEffect(() => {
    if (!historyLoaded) return;
    fetchAll();
    const id = setInterval(fetchAll, POLL_MS);
    return () => clearInterval(id);
  }, [historyLoaded]); // eslint-disable-line

  // ── Merged history for charts ──────────────────────────────────────────
  const maxLen = Math.max(...history.map(h => h.length), 0);
  const mergedHistory = Array.from({length: maxLen}, (_, i) => {
    const row = { time: history.find(h => h[i])?.[i]?.time ?? "" };
    SERVICES.forEach((svc, si) => {
      const pt = history[si][i];
      if (pt) {
        row[`${svc.name}_cpu`]  = pt.cpu;
        row[`${svc.name}_mem`]  = pt.memory;
        row[`${svc.name}_lat`]  = pt.latency;
        row[`${svc.name}_prob`] = pt.probability;
      }
    });
    return row;
  });

  const radarData = ["cpu","memory","error_rate"].map(key => ({
    metric: key==="error_rate" ? "Errors" : key.charAt(0).toUpperCase()+key.slice(1),
    ...Object.fromEntries(SERVICES.map((svc,i) => [svc.name, latest[i]?.[key] ?? 0]))
  }));

  const handleSimulate = async svc => {
    try {
      await axios.post(`/api/${svc.name.toLowerCase()}/simulate_failure`);
      addLog(`Failure injected → ${svc.name}`, "warn");
    } catch { addLog(`Could not inject failure on ${svc.name}`, "err"); }
  };

  const handleReset = async svc => {
    try {
      await axios.post(`/api/${svc.name.toLowerCase()}/reset`);
      addLog(`${svc.name} reset to healthy`, "ok");
    } catch { addLog(`Could not reset ${svc.name}`, "err"); }
  };

  const TABS = ["overview","cpu & memory","latency","radar"];

  return (
    <>
      <style>{`
        * { box-sizing:border-box; margin:0; padding:0; }
        body { background:#020817; color:#e2e8f0; font-family:'Inter',system-ui,sans-serif; }
        @keyframes flashBanner { from{opacity:1} to{opacity:.5} }
        @keyframes flashPill   { from{opacity:1} to{opacity:.3} }
        ::-webkit-scrollbar { width:4px; background:#0f172a; }
        ::-webkit-scrollbar-thumb { background:#1e293b; border-radius:2px; }
        input:focus { outline:1px solid #3b82f6; }
      `}</style>

      <AlertBanner alerts={alerts} onDismiss={() => setAlerts([])} />

      <div style={{maxWidth:1280,margin:"0 auto",padding:"28px 20px",paddingTop:alerts.length?80:28}}>

        {/* Header */}
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:28}}>
          <div>
            <h1 style={{fontSize:22,fontWeight:700,color:"#f8fafc"}}>Microservice Health Monitor</h1>
            <p style={{color:"#475569",fontSize:13,marginTop:3}}>
              Live prediction
            </p>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:8,
            background:"#0f172a",border:"1px solid #1e293b",
            borderRadius:20,padding:"5px 14px",fontSize:12}}>
            <span style={{width:7,height:7,borderRadius:"50%",background:"#22c55e",
              animation:"flashPill 1.4s ease-in-out infinite alternate",display:"inline-block"}} />
            <span style={{color:"#64748b"}}>Live</span>
          </div>
        </div>

        {/* Service Cards */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14,marginBottom:24}}>
          {SERVICES.map((svc,i) => (
            <ServiceCard key={svc.name} service={svc} latest={latest[i]}
              prediction={preds[i]} onSimulate={handleSimulate} onReset={handleReset} />
          ))}
        </div>

        {/* Charts */}
        <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,
          padding:"18px 20px",marginBottom:20}}>
          <div style={{display:"flex",gap:4,marginBottom:18}}>
            {TABS.map(t => (
              <button key={t} onClick={() => setActiveTab(t)} style={{
                background: activeTab===t ? "#1e3a5f" : "none",
                border:`1px solid ${activeTab===t ? "#1d4ed8" : "#1e293b"}`,
                color: activeTab===t ? "#93c5fd" : "#475569",
                borderRadius:6, padding:"5px 14px", fontSize:12, cursor:"pointer",
                fontWeight: activeTab===t ? 600 : 400, textTransform:"capitalize"
              }}>{t}</button>
            ))}
          </div>

          {activeTab==="overview" && (
            <>
              <div style={{color:"#64748b",fontSize:12,marginBottom:8}}>Failure probability % over time</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={mergedHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="time" tick={{fill:"#475569",fontSize:10}} interval="preserveStartEnd" />
                  <YAxis domain={[0,100]} tick={{fill:"#475569",fontSize:10}} />
                  <Tooltip contentStyle={{background:"#1e293b",border:"1px solid #334155",fontSize:12}} />
                  <Legend wrapperStyle={{fontSize:12}} />
                  {SERVICES.map(svc => (
                    <Line key={svc.name} type="monotone" dataKey={`${svc.name}_prob`}
                      name={svc.name} stroke={svc.color} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </>
          )}

          {activeTab==="cpu & memory" && (
            <>
              <div style={{color:"#64748b",fontSize:12,marginBottom:8}}>CPU % over time</div>
              <ResponsiveContainer width="100%" height={190}>
                <LineChart data={mergedHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="time" tick={{fill:"#475569",fontSize:10}} interval="preserveStartEnd" />
                  <YAxis domain={[0,100]} tick={{fill:"#475569",fontSize:10}} />
                  <Tooltip contentStyle={{background:"#1e293b",border:"1px solid #334155",fontSize:12}} />
                  <Legend wrapperStyle={{fontSize:12}} />
                  {SERVICES.map(svc => (
                    <Line key={svc.name} type="monotone" dataKey={`${svc.name}_cpu`}
                      name={`${svc.name} CPU`} stroke={svc.color} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div style={{color:"#64748b",fontSize:12,margin:"16px 0 8px"}}>Memory % — latest snapshot</div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={SERVICES.map((svc,i) => ({
                  name: svc.name,
                  memory: latest[i]?.memory ?? 0,
                  cpu:    latest[i]?.cpu ?? 0,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" tick={{fill:"#475569",fontSize:11}} />
                  <YAxis domain={[0,100]} tick={{fill:"#475569",fontSize:10}} />
                  <Tooltip contentStyle={{background:"#1e293b",border:"1px solid #334155",fontSize:12}} />
                  <Legend wrapperStyle={{fontSize:12}} />
                  <Bar dataKey="cpu"    name="CPU %"    fill="#38bdf8" radius={[3,3,0,0]} />
                  <Bar dataKey="memory" name="Memory %" fill="#a78bfa" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}

          {activeTab==="latency" && (
            <>
              <div style={{color:"#64748b",fontSize:12,marginBottom:8}}>Latency (ms) over time</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={mergedHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="time" tick={{fill:"#475569",fontSize:10}} interval="preserveStartEnd" />
                  <YAxis tick={{fill:"#475569",fontSize:10}} />
                  <Tooltip contentStyle={{background:"#1e293b",border:"1px solid #334155",fontSize:12}} />
                  <Legend wrapperStyle={{fontSize:12}} />
                  {SERVICES.map(svc => (
                    <Line key={svc.name} type="monotone" dataKey={`${svc.name}_lat`}
                      name={`${svc.name}`} stroke={svc.color} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </>
          )}

          {activeTab==="radar" && (
            <>
              <div style={{color:"#64748b",fontSize:12,marginBottom:8}}>CPU / Memory / Error Rate snapshot</div>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1e293b" />
                  <PolarAngleAxis dataKey="metric" tick={{fill:"#64748b",fontSize:12}} />
                  {SERVICES.map(svc => (
                    <Radar key={svc.name} name={svc.name} dataKey={svc.name}
                      stroke={svc.color} fill={svc.color} fillOpacity={0.15} />
                  ))}
                  <Legend wrapperStyle={{fontSize:12}} />
                  <Tooltip contentStyle={{background:"#1e293b",border:"1px solid #334155",fontSize:12}} />
                </RadarChart>
              </ResponsiveContainer>
            </>
          )}
        </div>

        {/* Model Input Drift Monitor (PSI) */}
        <DriftPanel driftInfo={driftInfo} />

        {/* Model Info + Anomaly Feed */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginBottom:14}}>
          <ModelInfoPanel modelInfo={modelInfo} featureImportance={featureImp} />
          <AnomalyFeed anomalies={anomalies} />
        </div>

        {/* Event Log + Email Manager */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <EventLog logs={logs} onClear={() => setLogs([])} />
          <EmailManager
            emails={emails}
            onAdd={e => {
              if (e && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e) && !emails.includes(e)) {
                setEmails(p => [...p, e]);
                axios.post("/api/alert-config/add", { email: e }).catch(()=>{});
              }
            }}
            onRemove={i => {
              const target = emails[i];
              setEmails(p => p.filter((_,j) => j!==i));
              if (target) {
                axios.post("/api/alert-config/remove", { email: target }).catch(()=>{});
              }
            }}
          />
        </div>
      </div>
    </>
  );
}
