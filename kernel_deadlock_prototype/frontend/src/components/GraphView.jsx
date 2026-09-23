import React, { useEffect, useRef } from "react";
import cytoscape from "cytoscape";

const STYLE = [
  {
    selector: "node[type='thread']",
    style: {
      shape: "ellipse",
      "background-color": "#3d8bfd",
      label: "data(label)",
      color: "#e6e8ec",
      "font-size": 9,
      "text-valign": "bottom",
      "text-margin-y": 4,
      width: 34,
      height: 34,
    },
  },
  {
    selector: "node[type='lock']",
    style: {
      shape: "round-rectangle",
      "background-color": "#8a91a3",
      label: "data(label)",
      color: "#e6e8ec",
      "font-size": 9,
      "text-valign": "bottom",
      "text-margin-y": 4,
      width: 30,
      height: 30,
    },
  },
  {
    selector: "node[waiting = 'true']",
    style: {
      "border-width": 3,
      "border-color": "#f5a623",
    },
  },
  {
    selector: "edge[type='owned_by']",
    style: {
      "line-color": "#2ecc71",
      "target-arrow-color": "#2ecc71",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      width: 2,
    },
  },
  {
    selector: "edge[type='waits_for']",
    style: {
      "line-color": "#f5a623",
      "target-arrow-color": "#f5a623",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      width: 2,
      "line-style": "dashed",
    },
  },
  {
    selector: "edge.cycle",
    style: {
      "line-color": "#e74c3c",
      "target-arrow-color": "#e74c3c",
      width: 4,
    },
  },
  {
    selector: "node.cycle",
    style: {
      "border-width": 4,
      "border-color": "#e74c3c",
    },
  },
];

/**
 * Detects directed cycles in the waits_for graph (thread -> lock chains
 * that loop back on themselves) so a real deadlock's dependency cycle
 * can be visually emphasized, per section 18 of the spec.
 */
function findCycleElementIds(elements) {
  const adjacency = new Map();
  for (const el of elements) {
    if (el.data.source && el.data.target) {
      if (!adjacency.has(el.data.source)) adjacency.set(el.data.source, []);
      adjacency.get(el.data.source).push({ to: el.data.target, edgeId: el.data.id });
    }
  }

  const visited = new Set();
  const stack = new Set();
  const cycleNodes = new Set();
  const cycleEdges = new Set();
  const path = [];

  function dfs(node) {
    visited.add(node);
    stack.add(node);
    path.push(node);

    for (const { to, edgeId } of adjacency.get(node) || []) {
      if (stack.has(to)) {
        // Found a cycle: mark the nodes/edges along path from `to` onward.
        const idx = path.indexOf(to);
        for (let i = idx; i < path.length; i++) cycleNodes.add(path[i]);
        cycleEdges.add(edgeId);
      } else if (!visited.has(to)) {
        dfs(to);
      }
    }

    stack.delete(node);
    path.pop();
  }

  for (const node of adjacency.keys()) {
    if (!visited.has(node)) dfs(node);
  }

  return { cycleNodes, cycleEdges };
}

function snapshotToElements(graph) {
  const nodes = (graph?.nodes || []).map((n) => ({
    data: {
      id: n.id,
      type: n.type,
      label: n.id,
      waiting: String(!!n.features?.is_waiting),
    },
  }));

  const edges = (graph?.edges || []).map((e, i) => ({
    data: {
      id: `${e.source}->${e.target}-${e.type}-${i}`,
      source: e.source,
      target: e.target,
      type: e.type,
    },
  }));

  return [...nodes, ...edges];
}

export default function GraphView({ graph }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  useEffect(() => {
    cyRef.current = cytoscape({
      container: containerRef.current,
      style: STYLE,
      layout: { name: "cose", animate: false },
      elements: [],
    });

    return () => cyRef.current?.destroy();
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !graph) return;

    const elements = snapshotToElements(graph);
    const { cycleNodes, cycleEdges } = findCycleElementIds(elements);

    cy.elements().remove();
    cy.add(elements);

    cy.nodes().forEach((n) => {
      if (cycleNodes.has(n.id())) n.addClass("cycle");
    });
    cy.edges().forEach((e) => {
      if (cycleEdges.has(e.id())) e.addClass("cycle");
    });

    cy.layout({ name: "cose", animate: false, fit: true, padding: 30 }).run();
  }, [graph]);

  return (
    <div className="panel graph-panel">
      <h2>Synchronization Graph (TSG)</h2>
      <div id="cy" ref={containerRef} />
    </div>
  );
}
