"""Model execution graph (architecture §9).

A DAG of model nodes; a single model is a one-node graph, a sequential
stack a chain, parallel-then-merge a fan-in. V1 executes in topological
order on one device via ONNX Runtime (MIGraphX/ROCm EP on AMD, CUDA EP on
the 3090 parity box); multi-HIP-stream overlap for independent branches
comes after single-stream e2e works.
"""
from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Any, Callable

import numpy as np

# Preference order per platform; first available wins.
_EP_PREFERENCE = [
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


def make_ort_session(model_path: str, device_ordinal: int = 0):
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list[Any] = []
    for ep in _EP_PREFERENCE:
        if ep in available:
            providers.append((ep, {"device_id": device_ordinal})
                             if ep != "CPUExecutionProvider" else ep)
    return ort.InferenceSession(model_path, providers=providers)


class OnnxModel:
    """Callable node body around an ORT session (single input, v1)."""

    def __init__(self, model_path: str, device_ordinal: int = 0,
                 postprocess: Callable | None = None):
        self.session = make_ort_session(model_path, device_ordinal)
        self.input_name = self.session.get_inputs()[0].name
        self.postprocess = postprocess

    def __call__(self, tensor: np.ndarray):
        outputs = self.session.run(None, {self.input_name: tensor})
        return self.postprocess(outputs) if self.postprocess else outputs


class ModelNode:
    def __init__(self, node_id: str, model: Callable, input_nodes: list[str] | None = None):
        self.node_id = node_id
        self.model = model
        self.input_nodes = input_nodes or []  # [] = takes the frame tensor


class ModelGraph:
    def __init__(self):
        self.nodes: dict[str, ModelNode] = {}

    def add_node(self, node_id: str, model: Callable,
                 input_nodes: list[str] | None = None) -> None:
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        for dep in input_nodes or []:
            if dep not in self.nodes:
                raise ValueError(f"node {node_id!r} depends on unknown node {dep!r}")
        self.nodes[node_id] = ModelNode(node_id, model, input_nodes)

    def topological_order(self) -> list[str]:
        ts = TopologicalSorter({nid: n.input_nodes for nid, n in self.nodes.items()})
        return list(ts.static_order())


class GraphExecutor:
    def __init__(self, graph: ModelGraph):
        self.graph = graph
        self.order = graph.topological_order()
        if not self.order:
            raise ValueError("empty model graph")

    def run(self, input_tensor) -> dict[str, Any]:
        """Returns every node's output keyed by node_id (callers usually
        want the detector's, not only the terminal node's)."""
        results: dict[str, Any] = {}
        for node_id in self.order:
            node = self.graph.nodes[node_id]
            inputs = ([input_tensor] if not node.input_nodes
                      else [results[d] for d in node.input_nodes])
            results[node_id] = node.model(*inputs)
        return results
