
from app.analysis_adapter import analysis_result_to_agent_result
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.verification.fact_checker import FactChecker


def test_analysis_result_reaches_knowledge_graph_and_fact_checker():
    raw_result = {
        "task_id": "T001",
        "task": "investigate_correlation",
        "columns": ["Pclass", "Fare"],
        "method": "Pearson correlation",
        "value": -0.549,
        "p_value": 0.0,
        "sample_size": 891,
    }

    agent_result = analysis_result_to_agent_result(
        raw_result,
        dataset="titanic.csv",
        confidence=0.9,  # Test fixture only, not a production policy.
    )

    assert agent_result.findings
    assert agent_result.evidence[0].finding_indices == [0]
    assert agent_result.features == ["Pclass", "Fare"]

    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    fact_checker = FactChecker(kg)

    finding_ids = builder.add_agent_result(agent_result)

    assert len(finding_ids) == 1

    verification = fact_checker.verify_finding(finding_ids[0])

    assert verification["verified"] is True
    assert verification["checks"]["has_evidence"] is True
    assert verification["checks"]["evidence_connected"] is True


def test_missingness_result_reaches_knowledge_graph_and_fact_checker():
    raw_result = {
        "task_id": "T001",
        "task": "investigate_missingness",
        "columns": ["Cabin"],
        "method": "Missing value count",
        "n_missing": 687,
        "pct_missing": 77.1,
        "sample_size": 891,
    }

    agent_result = analysis_result_to_agent_result(
        raw_result,
        dataset="titanic.csv",
        confidence=0.9,
    )

    kg = KnowledgeGraph()
    finding_ids = KnowledgeGraphBuilder(kg).add_agent_result(agent_result)
    verification = FactChecker(kg).verify_finding(finding_ids[0])

    assert verification["verified"] is True
    assert verification["checks"]["evidence_connected"] is True


def test_outlier_result_reaches_knowledge_graph_and_fact_checker():
    raw_result = {
        "task_id": "T002",
        "task": "investigate_outliers",
        "columns": ["Fare"],
        "method": "IQR outlier detection",
        "n_outliers": 116,
        "pct_outliers": 13.02,
        "bounds": [-26.72, 65.63],
        "sample_size": 891,
    }

    agent_result = analysis_result_to_agent_result(
        raw_result,
        dataset="titanic.csv",
        confidence=0.9,
    )

    kg = KnowledgeGraph()
    finding_ids = KnowledgeGraphBuilder(kg).add_agent_result(agent_result)
    verification = FactChecker(kg).verify_finding(finding_ids[0])

    assert verification["verified"] is True
    assert verification["checks"]["evidence_connected"] is True
