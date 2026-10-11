from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.reasoning import GraphReasoner
from app.schemas import AgentResult, Evidence, ModelResult


# Initialize the knowledge graph and its components
kg = KnowledgeGraph()
builder = KnowledgeGraphBuilder(kg)
reasoner = GraphReasoner(kg)


# Create sample agent output
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


# Add the result to the knowledge graph
finding_ids = builder.add_agent_result(result)

assert finding_ids, "No finding was created"

finding_id = finding_ids[0]


# Retrieve information from the graph
context = reasoner.get_finding_context(finding_id)
provenance = reasoner.get_provenance_chain(finding_id)
evidence = reasoner.get_supporting_evidence(finding_id)


# Display results
print("FINDING CONTEXT:")
print(context)

print("\nPROVENANCE CHAIN:")
print(provenance)

print("\nSUPPORTING EVIDENCE:")
print(evidence)


# Verify feature and dataset relationships
assert context["features"], "No supporting features found"
assert context["datasets"], "No dataset context found"

feature_names = [
    kg.get_node(feature_id).get("name")
    for feature_id in context["features"]
]

dataset_names = [
    kg.get_node(dataset_id).get("name")
    for dataset_id in context["datasets"]
]

assert "MonthlyCharges" in feature_names, (
    f"Expected MonthlyCharges, but found {feature_names}"
)

assert "customer_churn.csv" in dataset_names, (
    f"Expected customer_churn.csv, but found {dataset_names}"
)


# Verify evidence and model provenance
assert evidence, "No supporting evidence found"
assert provenance["models"], "No model provenance found"

print("\nBUILDER-REASONER INTEGRATION TEST PASSED")