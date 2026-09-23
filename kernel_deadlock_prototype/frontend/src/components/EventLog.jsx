import React from "react";

const EVENT_CLASS = {
  lock_wait_start: "ev-lock",
  lock_acquired: "ev-acquired",
  lock_released: "ev-released",
  lock_wait_timeout: "ev-lock",
};

export default function EventLog({ events }) {
  const rows = events && events.length ? events : [];

  return (
    <div className="panel event-log">
      <h2>Recent Events ({rows.length})</h2>
      {rows.length === 0 && <p>No events yet.</p>}
      {rows
        .slice()
        .reverse()
        .map((e, i) => (
          <div className="event-row" key={`${e.ts_ns}-${i}`}>
            <span className={EVENT_CLASS[e.event] || ""}>
              {String(e.event).padEnd(16, " ")}
            </span>{" "}
            tid={e.tid} lock={e.lock_id ?? "-"} cpu={e.cpu ?? "-"}
          </div>
        ))}
    </div>
  );
}
