import React from "react";

const STATE_LABELS = {
  safe: "SAFE",
  pre_deadlock: "PRE-DEADLOCK",
  deadlocked: "DEADLOCKED",
};

function pct(value) {
  return typeof value === "number" ? value.toFixed(4) : "—";
}

export default function PredictionPanel({ inference }) {
  if (!inference) {
    return (
      <div className="panel">
        <h2>Prediction</h2>
        <p>Waiting for backend...</p>
      </div>
    );
  }

  if (!inference.ready) {
    return (
      <div className="panel">
        <h2>Prediction</h2>
        <p>
          Warming up temporal window: {inference.snapshots ?? 0} /{" "}
          {inference.required ?? 8} snapshots
        </p>
      </div>
    );
  }

  const stateKey = inference.state || "safe";
  const thresholds = inference.thresholds || {};

  return (
    <div className="panel">
      <h2>Prediction</h2>
      <div className={`state-badge ${stateKey}`}>
        {STATE_LABELS[stateKey] || stateKey}
      </div>

      <div className="metric-row">
        <span>P(deadlock)</span>
        <span className="value">{pct(inference.deadlock_probability)}</span>
      </div>
      <div className="metric-row">
        <span>P(pre-deadlock)</span>
        <span className="value">{pct(inference.pre_deadlock_probability)}</span>
      </div>
      <div className="metric-row">
        <span>Risk ≤ 50 ms</span>
        <span className="value">{pct(inference.risk_50ms)}</span>
      </div>
      <div className="metric-row">
        <span>Risk ≤ 100 ms</span>
        <span className="value">{pct(inference.risk_100ms)}</span>
      </div>
      <div className="metric-row">
        <span>Risk ≤ 300 ms</span>
        <span className="value">{pct(inference.risk_300ms)}</span>
      </div>

      <div className="threshold-note">
        Thresholds (display only): deadlock ≥{" "}
        {thresholds.deadlock ?? "0.70"}, pre-deadlock ≥{" "}
        {thresholds.pre_deadlock ?? "0.56"}
      </div>
    </div>
  );
}
