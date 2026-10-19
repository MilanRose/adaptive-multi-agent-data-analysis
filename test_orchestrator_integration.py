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

        # Structural validation must pass for every finding.
        assert all(
            check["structural_verified"] is True
            for check in item["verification"]
        )

        # Scientific support is not assessed by the FactChecker.
        assert all(
            check["verified"] is False
            for check in item["verification"]
        )

        # Each result must include its execution evaluation.
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


def test_retrying_same_run_id_does_not_duplicate_graph_entries(
    tmp_path,
):
    graph_path = tmp_path / "knowledge_graph.json"

    orchestrator = Orchestrator(
        dataset_path="data/titanic.csv",
        graph_path=graph_path,
    )

    request = "Explore correlations and patterns in the Titanic dataset"
    run_id = "integration-retry-test-001"

    first = orchestrator.process_request(
        request,
        run_id=run_id,
    )

    graph = orchestrator.knowledge_graph.graph
    first_node_count = graph.number_of_nodes()
    first_edge_count = graph.number_of_edges()

    first_finding_ids = [
        item["finding_ids"] for item in first["results"]
    ]

    second = orchestrator.process_request(
        request,
        run_id=run_id,
    )

    assert first["run_id"] == run_id
    assert second["run_id"] == run_id

    assert [
        item["finding_ids"] for item in second["results"]
    ] == first_finding_ids

    assert graph.number_of_nodes() == first_node_count
    assert graph.number_of_edges() == first_edge_count

    execution_nodes = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "execution"
    ]

    assert len(execution_nodes) == len(first["results"])
    assert graph_path.exists()