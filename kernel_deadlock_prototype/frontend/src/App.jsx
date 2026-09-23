import React from "react";
import PredictionPanel from "./components/PredictionPanel.jsx";
import GraphView from "./components/GraphView.jsx";
import EventLog from "./components/EventLog.jsx";
import { useLiveSocket } from "./useLiveSocket.js";

export default function App() {
  const { connected, latest, eventLog, wsUrl } = useLiveSocket();

  return (
    <div className="app">
      <div className="header">
        <h1>Live Predictive Deadlock Detection</h1>
        <span className={`conn-status ${connected ? "connected" : "disconnected"}`}>
          {connected ? `connected (${wsUrl})` : `disconnected (${wsUrl})`}
        </span>
      </div>

      <div className="sidebar">
        <PredictionPanel inference={latest?.inference} />
        <EventLog events={eventLog} />
      </div>

      <div className="main-area">
        <GraphView graph={latest?.graph} />
      </div>
    </div>
  );
}
