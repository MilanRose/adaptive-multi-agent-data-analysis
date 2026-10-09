from app.knowledge_graph.nodes import (
    DatasetNode,
    FeatureNode,
    FindingNode,
    EvidenceNode,
    ModelNode
)


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


print(dataset.to_dict())
print(feature.to_dict())
print(finding.to_dict())
print(evidence.to_dict())
print(model.to_dict())