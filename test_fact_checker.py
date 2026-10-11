from app.schemas import AgentResult, Evidence, ModelResult
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.verification.fact_checker import FactChecker


# Create Knowledge Graph
kg = KnowledgeGraph()
builder = KnowledgeGraphBuilder(kg)


# Create a structured agent result
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
        metrics={"accuracy": 0.87}
    ),
    confidence=0.84
)


# Build Knowledge Graph
builder.add_agent_result(result)


# Find the generated finding ID
finding_id = next(
    node_id
    for node_id, data in kg.graph.nodes(data=True)
    if data.get("type") == "finding"
)


# Verify the finding
checker = FactChecker(kg)
verification = checker.verify_finding(finding_id)


# Display and check the result
print("\nFINDING ID:", finding_id)
print("\nVERIFICATION RESULT:")
print(verification)

assert verification["verified"] is True, (
    "The valid finding should pass verification."
)

print("\nTEST PASSED")