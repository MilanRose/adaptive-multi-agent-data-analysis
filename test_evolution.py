
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.schemas import AgentResult, Evidence, ModelResult


def create_result(agent, task, finding, evidence_description, confidence, accuracy):
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
            metrics={"accuracy": accuracy}
        ),
        confidence=confidence
    )


def main():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    # --------------------------------------------------
    # 1. Successful EDA execution
    # --------------------------------------------------
    result = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="MonthlyCharges is associated with churn",
        evidence_description="Pearson correlation = 0.42",
        confidence=0.84,
        accuracy=0.87
    )

    finding_ids = builder.add_agent_result(result)

    verification = {
        "verified": True,
        "checks": {
            "finding_exists": True,
            "has_confidence": True,
            "has_evidence": True,
            "has_model": True,
            "valid_confidence": True,
            "valid_model_accuracy": True
        }
    }

    evaluation = evolution.evaluate_execution(result, verification)
    execution_id = evolution.update_knowledge(
        result, evaluation, finding_ids
    )

    print("SUCCESSFUL EXECUTION:", execution_id)
    print("FINDINGS:", finding_ids)
    print("EXECUTION NODE:", kg.get_node(execution_id))

    assert kg.graph.has_edge(execution_id, finding_ids[0])
    assert kg.graph.edges[execution_id, finding_ids[0]]["relation"] == "produced"

    # --------------------------------------------------
    # 2. Failed ML execution
    # --------------------------------------------------
    failed_result = create_result(
        agent="MLAgent",
        task="correlation_analysis",
        finding="Test finding from MLAgent",
        evidence_description="Weak evidence",
        confidence=0.4,
        accuracy=0.4
    )

    failed_finding_ids = builder.add_agent_result(failed_result)

    failed_verification = {
        "verified": False,
        "checks": {
            "finding_exists": True,
            "has_confidence": True,
            "has_evidence": False,
            "has_model": True,
            "valid_confidence": True,
            "valid_model_accuracy": True
        }
    }

    failed_evaluation = evolution.evaluate_execution(
        failed_result, failed_verification
    )

    failed_execution_id = evolution.update_knowledge(
        failed_result, failed_evaluation, failed_finding_ids
    )

    print("\nFAILED EXECUTION:", failed_execution_id)
    print("FINDINGS:", failed_finding_ids)

    assert kg.graph.has_edge(failed_execution_id, failed_finding_ids[0])
    assert kg.graph.edges[failed_execution_id, failed_finding_ids[0]]["relation"] == "produced"

    # --------------------------------------------------
    # 3. Verify exact provenance
    # --------------------------------------------------
    assert not kg.graph.has_edge(execution_id, failed_finding_ids[0])
    assert not kg.graph.has_edge(failed_execution_id, finding_ids[0])

    print("\nSUCCESSFUL EXECUTIONS:")
    print(evolution.get_successful_executions())

    print("\nFAILED EXECUTIONS:")
    print(evolution.get_failed_executions())

    print("\nWORKFLOW RECOMMENDATION:")
    print(evolution.recommend_workflow("correlation_analysis"))

    print("\nPROVENANCE TEST PASSED")


if __name__ == "__main__":
    main()
