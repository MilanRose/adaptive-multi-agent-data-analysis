from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder


# Create Knowledge Graph
kg = KnowledgeGraph()

# Create Builder
builder = KnowledgeGraphBuilder(kg)


# Simulated result from EDA Agent
result = {

    "dataset_id": "dataset_1",
    "dataset": "customer_churn.csv",

    "feature_id": "feature_1",
    "feature": "MonthlyCharges",

    "finding_id": "finding_1",
    "finding": "MonthlyCharges is associated with churn",

    "confidence": 0.84,

    "evidence_id": "evidence_1",
    "evidence": "Pearson correlation = 0.42",

    "model_id": "model_1",
    "model": "Random Forest",
    "accuracy": 0.87
}


# Add agent result to Knowledge Graph
builder.add_agent_result(result)


# Display nodes
print("\nNODES:")

for node in kg.graph.nodes(data=True):
    print(node)


# Display relationships
print("\nRELATIONSHIPS:")

for edge in kg.graph.edges(data=True):
    print(edge)