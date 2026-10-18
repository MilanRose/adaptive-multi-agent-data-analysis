
from app.orchestrator.orchestrator import Orchestrator


def test_eda_request_runs_analysis_and_verifies_findings(tmp_path):
    graph_path = tmp_path / "knowledge_graph.json"

    orchestrator = Orchestrator(
        dataset_path="data/titanic.csv",
        graph_path=graph_path,
    )

    output = orchestrator.process_request(
        "Explore correlations and patterns in the Titanic dataset"
    )

    assert output["selected_agent"] == "EDAAgent"
    assert output["status"] == "completed"
    assert output["task_count"] == 9
    assert len(output["results"]) == 9
    assert graph_path.exists()

    for item in output["results"]:
        assert item["finding_ids"]
        assert item["verification"]
        assert all(
            check["verified"]
            for check in item["verification"]
        )
        assert "evaluation" in item

    # A fresh graph instance can reload the saved history.
    from app.knowledge_graph.graph import KnowledgeGraph

    reloaded_graph = KnowledgeGraph(storage_path=graph_path)

    assert any(
        attributes.get("type") == "execution"
        for _, attributes in reloaded_graph.graph.nodes(data=True)
    )


def test_unsupported_agent_has_consistent_response_fields(tmp_path):
    orchestrator = Orchestrator(
        graph_path=tmp_path / "knowledge_graph.json"
    )

    output = orchestrator.process_request(
        "Predict survival using a machine learning model"
    )

    assert output["selected_agent"] == "MLAgent"
    assert output["status"] == "not_implemented"
    assert output["dataset"] is None
    assert output["task_count"] == 0
    assert output["insights"] == []
    assert output["results"] == []
    assert output["errors"] == []
