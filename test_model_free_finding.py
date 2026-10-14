from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.schemas import AgentResult, Evidence
from app.verification.fact_checker import FactChecker


def test_model_free_finding():
    # Create knowledge graph and builder
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    # Create a valid finding with evidence but no ML model
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=[
            "MonthlyCharges is correlated with churn"
        ],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42,
                finding_indices=[0]
            )
        ],
        model=None,
        confidence=0.84
    )

    # Add the result to the knowledge graph
    finding_ids = builder.add_agent_result(result)

    assert len(finding_ids) == 1, "Exactly one finding should be created"

    finding_id = finding_ids[0]

    # Verify the finding
    checker = FactChecker(kg)
    verification = checker.verify_finding(finding_id)

    print("\nMODEL-FREE VERIFICATION RESULT:")
    print(verification)

    # A finding with valid evidence does not require an ML model
    assert verification["verified"] is True, (
        "A valid finding without an ML model should pass verification"
    )
    assert verification["checks"]["finding_exists"] is True
    assert verification["checks"]["has_confidence"] is True
    assert verification["checks"]["valid_confidence"] is True
    assert verification["checks"]["has_evidence"] is True
    assert verification["checks"]["evidence_connected"] is True
    assert verification["checks"]["has_model"] is False
    assert verification["checks"]["valid_model_accuracy"] is True

    print("\nMODEL-FREE TEST PASSED")


if __name__ == "__main__":
    test_model_free_finding()
