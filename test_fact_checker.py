from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.verification.fact_checker import FactChecker


# -------------------------
# Create Knowledge Graph
# -------------------------

kg = KnowledgeGraph()

builder = KnowledgeGraphBuilder(kg)


# -------------------------
# Simulated Agent Result
# -------------------------

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


# -------------------------
# Build Knowledge Graph
# -------------------------

builder.add_agent_result(result)


# -------------------------
# Fact Checker
# -------------------------

checker = FactChecker(kg)

verification = checker.verify_finding(
    "finding_1"
)


# -------------------------
# Display Result
# -------------------------

print("\nVERIFICATION RESULT:")

print(verification)