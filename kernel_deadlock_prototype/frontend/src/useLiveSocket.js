import { useEffect, useRef, useState } from "react";

// Single source of truth for the backend address (section 20 of the spec):
// set VITE_WS_URL in .env, never hardcode the VM IP elsewhere.
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";

/**
 * Connects to the live backend and keeps the latest message plus a
 * rolling event log. Auto-reconnects if the backend restarts, since the
 * monitoring loop is independent of any one frontend connection.
 */
export function useLiveSocket() {
  const [connected, setConnected] = useState(false);
  const [latest, setLatest] = useState(null);
  const [eventLog, setEventLog] = useState([]);
  const socketRef = useRef(null);
  const reconnectTimer = useRef(null);

  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;

      const ws = new WebSocket(WS_URL);
      socketRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (evt) => {
        try {
          const message = JSON.parse(evt.data);
          setLatest(message);
          if (Array.isArray(message.events) && message.events.length) {
            setEventLog(message.events);
          }
        } catch (err) {
          console.error("Failed to parse WS message", err);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          reconnectTimer.current = setTimeout(connect, 1500);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer.current);
      socketRef.current?.close();
    };
  }, []);

  return { connected, latest, eventLog, wsUrl: WS_URL };
}
