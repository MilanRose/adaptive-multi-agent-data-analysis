
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.schemas import AgentResult, Evidence
from app.verification.fact_checker import FactChecker


def test_model_free_finding():
    # Create knowledge graph and builder
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    # Create a valid result with evidence but NO model
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is correlated with churn"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42
            )
        ],
        model=None,
        confidence=0.84
    )

    # Add result to graph
    builder.add_agent_result(result)

    # Get the generated finding ID
    finding_ids = [
        node_id
        for node_id, data in kg.graph.nodes(data=True)
        if data.get("type") == "finding"
    ]

    assert len(finding_ids) > 0, "No finding was created"

    finding_id = finding_ids[0]

    # Verify the finding
    checker = FactChecker(kg)
    verification = checker.verify_finding(finding_id)

    print("\nMODEL-FREE VERIFICATION RESULT:")
    print(verification)

    assert verification["verified"] is True
    assert verification["checks"]["has_evidence"] is True
    assert verification["checks"]["has_model"] is False
    assert verification["checks"]["valid_model_accuracy"] is True

    print("\nMODEL-FREE TEST PASSED")


if __name__ == "__main__":
    test_model_free_finding()