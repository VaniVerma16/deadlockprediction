from __future__ import annotations

import math
import pathlib
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "gnn_v5"
    / "best_model_v5.pt"
)

RELATIONS = ("owned_by", "waits_for")

LABELS = (
    "safe",
    "pre_deadlock",
    "deadlocked",
)


# ---------------------------------------------------------------------
# V5 node features
# ---------------------------------------------------------------------

def node_features(node: dict[str, Any]) -> list[float]:
    """
    Exact 9-D feature contract used by V5.

    0: is_thread
    1: is_lock
    2: is_waiting
    3: log1p(wait_ns) / 22
    4: has_owner
    5: log1p(scheduler_switches) / 8
    6: log1p(wakeups) / 8
    7: log1p(cpu_migrations) / 6
    8: normalized last_cpu
    """

    node_type = node.get("type", "")
    features = node.get("features", {})

    return [
        float(node_type == "thread"),
        float(node_type == "lock"),
        float(features.get("is_waiting", 0)),
        math.log1p(
            float(features.get("wait_ns", 0))
        ) / 22.0,
        float(features.get("has_owner", 0)),
        math.log1p(
            float(features.get("scheduler_switches", 0))
        ) / 8.0,
        math.log1p(
            float(features.get("wakeups", 0))
        ) / 8.0,
        math.log1p(
            float(features.get("cpu_migrations", 0))
        ) / 6.0,
        float(features.get("last_cpu", 0)) / 7.0,
    ]


# ---------------------------------------------------------------------
# Graph tensorization
# ---------------------------------------------------------------------

def tensorize_graph(
    snapshot: dict[str, Any],
) -> dict[str, Any]:

    nodes = snapshot["nodes"]

    node_index = {
        node["id"]: i
        for i, node in enumerate(nodes)
    }

    x = torch.tensor(
        [
            node_features(node)
            for node in nodes
        ],
        dtype=torch.float32,
    )

    edge_lists = {
        relation: []
        for relation in RELATIONS
    }

    for edge in snapshot.get("edges", []):

        relation = edge["type"]

        if relation not in edge_lists:
            continue

        source = node_index[edge["source"]]
        target = node_index[edge["target"]]

        edge_lists[relation].append(
            (source, target)
        )

    edges = {}

    for relation in RELATIONS:

        pairs = edge_lists[relation]

        if pairs:
            edges[relation] = torch.tensor(
                pairs,
                dtype=torch.long,
            ).t().contiguous()

        else:
            edges[relation] = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

    return {
        "x": x,
        "edges": edges,
    }


# ---------------------------------------------------------------------
# Relational GraphSAGE
# ---------------------------------------------------------------------

class RelationalGraphSAGE(nn.Module):

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 32,
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            input_size,
            hidden_size,
        )

        self.self_layer = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.relation_layers = nn.ModuleDict(
            {
                relation: nn.Linear(
                    hidden_size,
                    hidden_size,
                    bias=False,
                )
                for relation in RELATIONS
            }
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

    def aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        transform: nn.Module,
    ) -> torch.Tensor:

        if edge_index.numel() == 0:
            return torch.zeros_like(h)

        source = edge_index[0]
        target = edge_index[1]

        messages = transform(
            h[source]
        )

        aggregated = torch.zeros_like(h)

        aggregated.index_add_(
            0,
            target,
            messages,
        )

        counts = torch.zeros(
            h.size(0),
            device=h.device,
            dtype=h.dtype,
        )

        counts.index_add_(
            0,
            target,
            torch.ones(
                target.size(0),
                device=h.device,
                dtype=h.dtype,
            ),
        )

        counts = counts.clamp_min(
            1.0
        ).unsqueeze(1)

        return aggregated / counts

    def forward(
        self,
        graph: dict[str, Any],
    ) -> torch.Tensor:

        x = graph["x"]
        edges = graph["edges"]

        h = F.relu(
            self.input_proj(x)
        )

        out = self.self_layer(h)

        for relation in RELATIONS:

            out = out + self.aggregate(
                h,
                edges[relation],
                self.relation_layers[relation],
            )

        h = F.relu(
            self.norm(out)
        )

        thread_mask = x[:, 0] > 0.5
        lock_mask = x[:, 1] > 0.5

        if thread_mask.any():
            thread_pool = h[
                thread_mask
            ].mean(dim=0)
        else:
            thread_pool = h.mean(dim=0)

        if lock_mask.any():
            lock_pool = h[
                lock_mask
            ].mean(dim=0)
        else:
            lock_pool = h.mean(dim=0)

        return torch.cat(
            [
                thread_pool,
                lock_pool,
            ],
            dim=0,
        )


# ---------------------------------------------------------------------
# V5 model
# ---------------------------------------------------------------------

class TemporalRiskAwareV5(nn.Module):

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 32,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.gnn = RelationalGraphSAGE(
            input_size=input_size,
            hidden_size=hidden_size,
        )

        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.deadlock_head = nn.Linear(
            hidden_size,
            1,
        )

        self.pre_head = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_50_head = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_100_head = nn.Linear(
            hidden_size,
            1,
        )

        self.risk_300_head = nn.Linear(
            hidden_size,
            1,
        )

    def encode(
        self,
        batch_graph_sequences,
        device,
    ):

        sequence_embeddings = []

        for graphs in batch_graph_sequences:

            embeddings = []

            for graph in graphs:

                graph = {
                    "x": graph["x"].to(device),
                    "edges": {
                        relation: edge.to(device)
                        for relation, edge
                        in graph["edges"].items()
                    },
                }

                embeddings.append(
                    self.gnn(graph)
                )

            sequence_embeddings.append(
                torch.stack(embeddings)
            )

        x = torch.stack(
            sequence_embeddings,
            dim=0,
        )

        output, _ = self.gru(x)

        return self.norm(
            output[:, -1, :]
        )

    def forward(
        self,
        batch_graph_sequences,
        device,
    ):

        h = self.encode(
            batch_graph_sequences,
            device,
        )

        return {
            "deadlock": self.deadlock_head(h).squeeze(1),
            "pre": self.pre_head(h).squeeze(1),
            "risk_50": self.risk_50_head(h).squeeze(1),
            "risk_100": self.risk_100_head(h).squeeze(1),
            "risk_300": self.risk_300_head(h).squeeze(1),
        }


# ---------------------------------------------------------------------
# V5 inference wrapper
# ---------------------------------------------------------------------

class V5Inference:

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ):
        self.model_path = Path(model_path)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # The checkpoint was originally created on Windows.
        pathlib.WindowsPath = pathlib.PosixPath

        print(
            f"Loading V5 model from:\n"
            f"  {self.model_path}"
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        config = checkpoint["config"]

        self.hidden_size = config[
            "hidden_size"
        ]

        self.deadlock_threshold = float(
            config["deadlock_threshold"]
        )

        self.pre_threshold = float(
            config["pre_threshold"]
        )

        self.model = TemporalRiskAwareV5(
            input_size=9,
            hidden_size=self.hidden_size,
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # V5 uses 8 snapshots per temporal sequence.
        self.window = deque(
            maxlen=8
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Hidden size: "
            f"{self.hidden_size}"
        )

        print(
            f"Deadlock threshold: "
            f"{self.deadlock_threshold:.2f}"
        )

        print(
            f"Pre-deadlock threshold: "
            f"{self.pre_threshold:.2f}"
        )

        print(
            "✓ V5 model loaded"
        )

    def add_snapshot(
        self,
        snapshot: dict[str, Any],
    ):

        self.window.append(
            tensorize_graph(snapshot)
        )

        if len(self.window) < 8:

            return {
                "ready": False,
                "snapshots": len(self.window),
                "required": 8,
            }

        return self.predict()

    @torch.no_grad()
    def predict(self):

        graphs = list(self.window)

        outputs = self.model(
            [graphs],
            self.device,
        )

        deadlock_probability = torch.sigmoid(
            outputs["deadlock"]
        ).item()

        pre_probability = torch.sigmoid(
            outputs["pre"]
        ).item()

        risk_50 = torch.sigmoid(
            outputs["risk_50"]
        ).item()

        risk_100 = torch.sigmoid(
            outputs["risk_100"]
        ).item()

        risk_300 = torch.sigmoid(
            outputs["risk_300"]
        ).item()

        if (
            deadlock_probability
            >= self.deadlock_threshold
        ):

            state = "deadlocked"

        elif (
            pre_probability
            >= self.pre_threshold
        ):

            state = "pre_deadlock"

        else:

            state = "safe"

        return {
            "ready": True,
            "state": state,
            "deadlock_probability": deadlock_probability,
            "pre_deadlock_probability": pre_probability,
            "risk_50ms": risk_50,
            "risk_100ms": risk_100,
            "risk_300ms": risk_300,
            "thresholds": {
                "deadlock": self.deadlock_threshold,
                "pre_deadlock": self.pre_threshold,
            },
        }