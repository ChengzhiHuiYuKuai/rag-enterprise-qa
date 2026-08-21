"""LangGraph 状态图测试"""

from app.core.graph import build_rag_graph


def test_graph_structure():
    """测试状态图结构是否正确构建"""
    graph = build_rag_graph()

    # 验证图可以被编译
    assert graph is not None

    # 验证图的节点（精简版：retrieve → generate）
    nodes = graph.get_graph().nodes
    expected_nodes = {"retrieve", "generate"}
    assert expected_nodes.issubset(set(nodes.keys()))
