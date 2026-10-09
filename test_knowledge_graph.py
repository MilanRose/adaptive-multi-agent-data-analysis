from app.knowledge_graph.graph import KnowledgeGraph

from app.knowledge_graph.nodes import (
    DatasetNode,
    FeatureNode,
    FindingNode,
    EvidenceNode,
    ModelNode
)

from app.knowledge_graph.reasoning import GraphReasoner


# -------------------------
# Create Knowledge Graph
# -------------------------

kg = KnowledgeGraph()


# -------------------------
# Create Nodes
# -------------------------

dataset = DatasetNode(
    "dataset_1",
    "customer_churn.csv"
)

feature = FeatureNode(
    "feature_1",
    "MonthlyCharges"
)

finding = FindingNode(
    "finding_1",
    "MonthlyCharges is associated with churn",
    confidence=0.84
)

evidence = EvidenceNode(
    "evidence_1",
    "Pearson correlation = 0.42",
    method="Pearson correlation"
)

model = ModelNode(
    "model_1",
    "Random Forest",
    accuracy=0.87
)


# -------------------------
# Add Nodes to Graph
# -------------------------

kg.add_node_object(dataset)
kg.add_node_object(feature)
kg.add_node_object(finding)
kg.add_node_object(evidence)
kg.add_node_object(model)


# -------------------------
# Add Relationships
# -------------------------

kg.add_relationship(
    "dataset_1",
    "contains",
    "feature_1"
)

kg.add_relationship(
    "feature_1",
    "supports",
    "finding_1"
)

kg.add_relationship(
    "finding_1",
    "supported_by",
    "evidence_1"
)

kg.add_relationship(
    "finding_1",
    "generated_by",
    "model_1"
)


# -------------------------
# Display Graph
# -------------------------

print("\nNODES:")

for node in kg.graph.nodes(data=True):
    print(node)


print("\nRELATIONSHIPS:")

for edge in kg.graph.edges(data=True):
    print(edge)


# -------------------------
# Graph Reasoning
# -------------------------

reasoner = GraphReasoner(kg)

evidence_results = reasoner.get_supporting_evidence(
    "finding_1"
)

print("\nSUPPORTING EVIDENCE:")

for item in evidence_results:
    print(item)
    
context = reasoner.get_finding_context(
    "finding_1"
)

print("\nFINDING CONTEXT:")

print(context)

# -------------------------
# Provenance Chain
# -------------------------

provenance = reasoner.get_provenance_chain(
    "finding_1"
)

print("\nPROVENANCE CHAIN:")

print(provenance)