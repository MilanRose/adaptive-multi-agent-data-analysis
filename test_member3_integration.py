
from app.schemas import AgentResult, Evidence, ModelResult
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.verification.fact_checker import FactChecker
from app.knowledge_graph.reasoning import GraphReasoner


def main():
    # 1. Create the knowledge graph and components
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    fact_checker = FactChecker(kg)
    evolution = SelfEvolutionEngine(kg)
    reasoner = GraphReasoner(kg)

    # 2. Create an agent result
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        features=["MonthlyCharges"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": 0.87}
        ),
        confidence=0.84
    )

    # 3. Build the graph
    finding_ids = builder.add_agent_result(result)
    assert finding_ids, "No finding IDs were created"

    finding_id = finding_ids[0]
    print("\nFINDING ID:", finding_id)

    # 4. Verify the finding
    verification = fact_checker.verify_finding(finding_id)
    print("\nFACT CHECK RESULT:")
    print(verification)

    assert verification["verified"] is True

    # 5. Check finding context and provenance
    context = reasoner.get_finding_context(finding_id)
    provenance = reasoner.get_provenance_chain(finding_id)

    assert context["features"], "Finding has no feature context"
    assert context["datasets"], "Finding has no dataset context"
    assert provenance["evidence"], "Finding has no evidence provenance"
    assert provenance["models"], "Finding has no model provenance"

    # 6. Evaluate the execution
    evaluation = evolution.evaluate_execution(result, verification)
    print("\nEXECUTION EVALUATION:")
    print(evaluation)

    assert evaluation["successful"] is True

    # 7. Store execution history
    execution_id = evolution.update_knowledge(
        result,
        evaluation,
        finding_ids
    )

    assert execution_id in kg.graph.nodes

    # Verify the produced relationship with MultiDiGraph-compatible access
    produced_edge_exists = any(
        target == finding_id
        and data.get("relation") == "produced"
        for _, target, data in kg.graph.out_edges(
            execution_id,
            data=True
        )
    )

    assert produced_edge_exists, (
        "Execution is not linked to its finding by a produced relationship"
    )

    print("\nEXECUTION STORED:", execution_id)

    # 8. Check execution history and workflow recommendation
    successful = evolution.get_successful_executions()
    assert any(
        item["execution_id"] == execution_id
        for item in successful
    ), "Stored execution was not found in successful history"

    recommendation = evolution.recommend_workflow("correlation_analysis")
    print("\nWORKFLOW RECOMMENDATION:")
    print(recommendation)

    assert (
        evolution.get_recommended_agent("correlation_analysis")
        == "EDAAgent"
    )

    print("\nALL INTEGRATION ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
