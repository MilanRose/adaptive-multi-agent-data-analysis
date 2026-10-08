from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.schemas import AgentResult, Evidence, ModelResult


kg = KnowledgeGraph()

evolution = SelfEvolutionEngine(kg)


result = AgentResult(
    agent="EDAAgent",
    dataset="customer_churn.csv",
    task="correlation_analysis",
    findings=[
        "MonthlyCharges is associated with churn"
    ],
    evidence=[
        Evidence(
            description="Pearson correlation = 0.42",
            method="Pearson correlation",
            value=0.42
        )
    ],
    model=ModelResult(
        name="Random Forest",
        metrics={
            "accuracy": 0.87
        }
    ),
    confidence=0.84
)


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


evaluation = evolution.evaluate_execution(
    result,
    verification
)


print("EXECUTION EVALUATION:")
print(evaluation)

execution_id = evolution.update_knowledge(
    result,
    evaluation
)

print("EXECUTION STORED:")
print(execution_id)

print("EXECUTION NODE:")
print(kg.get_node(execution_id))

successful = evolution.get_successful_executions()

print("SUCCESSFUL EXECUTIONS:")
print(successful)

similar = evolution.find_similar_executions(
    "correlation_analysis"
)

print("SIMILAR EXECUTIONS:")
print(similar)

recommendation = evolution.recommend_workflow(
    "correlation_analysis"
)

print("WORKFLOW RECOMMENDATION:")
print(recommendation)

recommended_agent = evolution.get_recommended_agent(
    "correlation_analysis"
)

print("RECOMMENDED AGENT:")
print(recommended_agent)

failed_result = AgentResult(
    agent="MLAgent",
    dataset="customer_churn.csv",
    task="correlation_analysis",
    findings=[
        "Test finding"
    ],
    evidence=[
        Evidence(
            description="Weak evidence",
            method="test",
            value=0.2
        )
    ],
    model=ModelResult(
        name="Test Model",
        metrics={
            "accuracy": 0.4
        }
    ),
    confidence=0.4
)

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
    failed_result,
    failed_verification
)

print("FAILED EXECUTION EVALUATION:")
print(failed_evaluation)
failed_execution_id = evolution.update_knowledge(
    failed_result,
    failed_evaluation
)

print("FAILED EXECUTION STORED:")
print(failed_execution_id)

print("FAILED EXECUTION NODE:")
print(kg.get_node(failed_execution_id))

failed = evolution.get_failed_executions()

print("FAILED EXECUTIONS:")
print(failed)

print("\nFINAL WORKFLOW RECOMMENDATION:")

final_recommendation = evolution.recommend_workflow(
    "correlation_analysis"
)

print(final_recommendation)

print("\nFINAL RECOMMENDED AGENT:")

final_agent = evolution.get_recommended_agent(
    "correlation_analysis"
)

print(final_agent)