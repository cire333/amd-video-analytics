import pytest

from avap.graph import GraphExecutor, ModelGraph


def test_single_node():
    g = ModelGraph()
    g.add_node("det", lambda x: x * 2)
    assert GraphExecutor(g).run(21)["det"] == 42


def test_sequential_chain():
    g = ModelGraph()
    g.add_node("detector", lambda x: x + 1)
    g.add_node("classifier", lambda x: x * 10, input_nodes=["detector"])
    results = GraphExecutor(g).run(1)
    assert results["detector"] == 2
    assert results["classifier"] == 20


def test_parallel_then_bridge():
    g = ModelGraph()
    g.add_node("rgb", lambda x: x + 1)
    g.add_node("thermal", lambda x: x + 2)
    g.add_node("fusion", lambda a, b: (a, b), input_nodes=["rgb", "thermal"])
    results = GraphExecutor(g).run(0)
    assert results["fusion"] == (1, 2)


def test_unknown_dependency_rejected():
    g = ModelGraph()
    with pytest.raises(ValueError, match="unknown node"):
        g.add_node("classifier", lambda x: x, input_nodes=["missing"])


def test_duplicate_node_rejected():
    g = ModelGraph()
    g.add_node("det", lambda x: x)
    with pytest.raises(ValueError, match="duplicate"):
        g.add_node("det", lambda x: x)
