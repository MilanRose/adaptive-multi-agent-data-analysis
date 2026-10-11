
from app.schemas import AgentResult, Evidence, ModelResult
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder


def create_result(agent, task, finding, evidence_description, confidence):
    return AgentResult(
        agent=agent,
        dataset="customer_churn.csv",
        task=task,
        findings=[finding],
        evidence=[
            Evidence(
                description=evidence_description,
                method="Test evidence",
                value=0.42
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": 0.87}
        ),
        confidence=confidence
    )


def main():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    # First independent result
    result1 = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="MonthlyCharges is associated with churn",
        evidence_description="Pearson correlation = 0.42",
        confidence=0.84
    )

    created_finding_ids_1 = builder.add_agent_result(result1)
    print("First result findings:", created_finding_ids_1)

    # Second independent result
    result2 = create_result(
        agent="MLAgent",
        task="classification",
        finding="Random Forest predicts customer churn",
        evidence_description="Random Forest accuracy = 0.87",
        confidence=0.81
    )

    created_finding_ids_2 = builder.add_agent_result(result2)
    print("Second result findings:", created_finding_ids_2)

    # Verify the builder returns the correct finding IDs
    assert created_finding_ids_1 == ["finding_1_1"]
    assert created_finding_ids_2 == ["finding_2_1"]

    print("\nNODES:")
    for node_id, data in kg.graph.nodes(data=True):
        print(node_id, data)

    print("\nRELATIONSHIPS:")
    for source, target, data in kg.graph.edges(data=True):
        print(source, "->", target, data)

    print("\nBUILDER TEST PASSED")


if __name__ == "__main__":
    main()