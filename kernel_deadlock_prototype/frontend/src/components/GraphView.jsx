import React, { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";

const STYLE = [
  { selector: "node", style: { label: "data(label)", color: "#dbe4f0", "font-size": 11, "font-family": "system-ui", "text-valign": "bottom", "text-margin-y": 8, "text-outline-width": 2, "text-outline-color": "#101722", "overlay-opacity": 0 } },
  { selector: "node[type='thread']", style: { shape: "ellipse", "background-color": "#4a92e5", "border-color": "#86bcfa", "border-width": 1, width: 36, height: 36 } },
  { selector: "node[type='lock']", style: { shape: "round-rectangle", "background-color": "#68758a", "border-color": "#a3afc0", "border-width": 1, width: 32, height: 32 } },
  { selector: "node[waiting='true']", style: { "border-width": 3, "border-color": "#f8bc61" } },
  { selector: "edge", style: { width: 2, "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.85, opacity: 0.85 } },
  { selector: "edge[type='owned_by']", style: { "line-color": "#49b98c", "target-arrow-color": "#49b98c" } },
  { selector: "edge[type='waits_for']", style: { "line-color": "#f8bc61", "target-arrow-color": "#f8bc61", "line-style": "dashed" } },
  { selector: "edge.cycle", style: { "line-color": "#ff657a", "target-arrow-color": "#ff657a", "line-style": "solid", width: 4, opacity: 1, "z-index": 20 } },
  { selector: "node.cycle", style: { "border-width": 4, "border-color": "#ff657a" } },
  { selector: ":selected", style: { "border-color": "#ffffff", "border-width": 3, opacity: 1 } },
];

function snapshotToElements(graph) {
  const nodes = (graph?.nodes || []).map((n) => ({
    data: { id: String(n.id), type: n.type, label: String(n.id), waiting: String(!!n.features?.is_waiting) },
  }));
  const counts = new Map();
  const edges = (graph?.edges || []).map((e) => {
    const signature = [e.source, e.target, e.type].join("|");
    const ordinal = counts.get(signature) || 0;
    counts.set(signature, ordinal + 1);
    return { data: { id: "e:" + signature + ":" + ordinal, source: String(e.source), target: String(e.target), type: e.type } };
  });
  return [...nodes, ...edges];
}

// Grey/amber edges can belong to a directed cycle. Only the actual cycle is red.
function findCycleElementIds(elements) {
  const adjacency = new Map();
  for (const el of elements) {
    const { source, target, id } = el.data;
    if (!source || !target) continue;
    if (!adjacency.has(source)) adjacency.set(source, []);
    adjacency.get(source).push({ to: target, edgeId: id });
  }
  const visited = new Set(), active = new Map(), pathNodes = [], pathEdges = [];
  const cycleNodes = new Set(), cycleEdges = new Set();
  function dfs(node) {
    visited.add(node);
    active.set(node, pathNodes.length);
    pathNodes.push(node);
    for (const { to, edgeId } of adjacency.get(node) || []) {
      if (active.has(to)) {
        const start = active.get(to);
        for (let i = start; i < pathNodes.length; i++) cycleNodes.add(pathNodes[i]);
        for (let i = start; i < pathEdges.length; i++) cycleEdges.add(pathEdges[i]);
        cycleEdges.add(edgeId);
      } else if (!visited.has(to)) {
        pathEdges.push(edgeId);
        dfs(to);
        pathEdges.pop();
      }
    }
    pathNodes.pop();
    active.delete(node);
  }
  for (const node of adjacency.keys()) if (!visited.has(node)) dfs(node);
  return { cycleNodes, cycleEdges };
}

export default function GraphView({ graph }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const initializedRef = useRef(false);
  const layoutRef = useRef(null);
  const [counts, setCounts] = useState({ threads: 0, locks: 0, waits: 0, cycles: 0 });
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  const lastGraphRef = useRef(null);
  const applyRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current, style: STYLE, elements: [],
      layout: { name: "preset" }, minZoom: 0.2, maxZoom: 3,
      wheelSensitivity: 0.18,
      boxSelectionEnabled: false,
    });
    cyRef.current = cy;
    const observer = new ResizeObserver(() => cy.resize());
    observer.observe(containerRef.current);
    return () => {
      observer.disconnect();
      layoutRef.current?.stop();
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  applyRef.current = (incoming) => {
    const cy = cyRef.current;
    if (!cy || !incoming) return;
    const elements = snapshotToElements(incoming);
    const { cycleNodes, cycleEdges } = findCycleElementIds(elements);
    const ids = new Set(elements.map((el) => el.data.id));
    const existingNodeIds = new Set(cy.nodes().map((n) => n.id()));
    const nextNodeIds = new Set(elements.filter((el) => !el.data.source).map((el) => el.data.id));
    const topologyChanged = [...nextNodeIds].some((id) => !existingNodeIds.has(id)) ||
      [...existingNodeIds].some((id) => !nextNodeIds.has(id));

    cy.batch(() => {
      cy.elements().filter((el) => !ids.has(el.id())).remove();
      // Insert nodes first so that new edges always have valid endpoints.
      for (const el of elements.filter((item) => !item.data.source)) {
        const current = cy.getElementById(el.data.id);
        if (current.length) current.data(el.data);
        else cy.add(el);
      }
      for (const el of elements.filter((item) => item.data.source)) {
        const current = cy.getElementById(el.data.id);
        if (current.length) current.data(el.data);
        else if (cy.getElementById(el.data.source).length && cy.getElementById(el.data.target).length) cy.add(el);
      }
      cy.elements().removeClass("cycle");
      cy.nodes().forEach((node) => { if (cycleNodes.has(node.id())) node.addClass("cycle"); });
      cy.edges().forEach((edge) => { if (cycleEdges.has(edge.id())) edge.addClass("cycle"); });
    });

    setCounts({
      threads: elements.filter((el) => el.data.type === "thread").length,
      locks: elements.filter((el) => el.data.type === "lock").length,
      waits: elements.filter((el) => el.data.type === "waits_for").length,
      cycles: cycleNodes.size,
    });

    if (topologyChanged || !initializedRef.current) {
      // A layout is expensive and disorienting on every frame. Run only when
      // nodes appear/disappear, and preserve existing node coordinates.
      layoutRef.current?.stop();
      const first = !initializedRef.current;
      initializedRef.current = true;
      const anchors = new Map(cy.nodes().filter((n) => existingNodeIds.has(n.id())).map((n) => [n.id(), { ...n.position() }]));
      const layout = cy.layout({
        name: "cose", animate: false, fit: first, padding: 60,
        randomize: first, nodeRepulsion: () => 12000,
        idealEdgeLength: () => 120, numIter: 180,
      });
      layoutRef.current = layout;
      layout.run();
      if (!first) {
        cy.batch(() => { cy.nodes().forEach((n) => { if (anchors.has(n.id())) n.position(anchors.get(n.id())); }); });
      }
    }
  };

  useEffect(() => {
    if (!graph) return;
    lastGraphRef.current = graph;
    if (!pausedRef.current) applyRef.current?.(graph);
  }, [graph]);

  const togglePause = () => {
    const next = !pausedRef.current;
    pausedRef.current = next;
    setPaused(next);
    if (!next) applyRef.current?.(lastGraphRef.current);
  };

  const resetLayout = () => {
    const cy = cyRef.current;
    if (!cy || cy.nodes().empty()) return;
    layoutRef.current?.stop();
    layoutRef.current = cy.layout({ name: "cose", animate: false, fit: true, padding: 60, nodeRepulsion: () => 12000, idealEdgeLength: () => 120, numIter: 200 });
    layoutRef.current.run();
  };

  return (
    <div className="panel graph-panel">
      <div className="graph-heading">
        <div><h2>Synchronization Graph</h2><p>Live thread and lock dependencies</p></div>
        <div className="graph-actions">
          <button type="button" onClick={togglePause}>{paused ? "Resume" : "Pause"}</button>
          <button type="button" onClick={() => cyRef.current?.fit(undefined, 50)}>Fit</button>
          <button type="button" onClick={resetLayout}>Reset layout</button>
        </div>
      </div>
      <div className="graph-meta">
        <span>{counts.threads} threads</span><span>{counts.locks} locks</span><span>{counts.waits} waiting edges</span>
        {counts.cycles > 0 && <strong>{counts.cycles} cycle nodes</strong>}
        {paused && <strong>View paused</strong>}
      </div>
      <div className="graph-canvas">
        <div id="cy" ref={containerRef} />
        {!graph && <div className="graph-empty">Waiting for synchronization data…</div>}
      </div>
      <div className="graph-legend">
        <span><i className="legend-dot thread" />Thread</span>
        <span><i className="legend-dot lock" />Lock</span>
        <span><i className="legend-line owned" />Owned by</span>
        <span><i className="legend-line waiting" />Waits for</span>
        <span><i className="legend-line cycle" />Cycle</span>
        <span className="graph-tip">Drag nodes to inspect · Scroll to zoom</span>
      </div>
    </div>
  );
}
