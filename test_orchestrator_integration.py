
from app.orchestrator.orchestrator import Orchestrator


def test_eda_request_runs_analysis_and_verifies_findings():
    orchestrator = Orchestrator(dataset_path="data/titanic.csv")

    output = orchestrator.process_request(
        "Explore correlations and patterns in the Titanic dataset"
    )

    assert output["selected_agent"] == "EDAAgent"
    assert output["status"] == "completed"
    assert output["task_count"] == 9
    assert len(output["results"]) == 9

    for item in output["results"]:
        assert item["finding_ids"]
        assert item["verification"]
        assert all(
            check["verified"]
            for check in item["verification"]
        )


def test_unsupported_agent_is_not_reported_as_executed():
    orchestrator = Orchestrator()

    output = orchestrator.process_request(
        "Predict survival using a machine learning model"
    )

    assert output["selected_agent"] == "MLAgent"
    assert output["status"] == "not_implemented"
    assert output["results"] == []
