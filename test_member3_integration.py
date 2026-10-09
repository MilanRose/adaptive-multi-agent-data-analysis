from app.schemas import AgentResult, Evidence, ModelResult
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.verification.fact_checker import FactChecker
from app.knowledge_graph.reasoning import GraphReasoner


# =========================================================
# 1. CREATE KNOWLEDGE GRAPH
# =========================================================

kg = KnowledgeGraph()

builder = KnowledgeGraphBuilder(kg)

fact_checker = FactChecker(kg)

evolution = SelfEvolutionEngine(kg)


# =========================================================
# 2. CREATE AGENT RESULT
# =========================================================

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


# =========================================================
# 3. BUILD KNOWLEDGE GRAPH
# =========================================================

builder.add_agent_result(result)


print("\n========================================")
print("KNOWLEDGE GRAPH CREATED")
print("========================================")

for node in kg.graph.nodes(data=True):
    print(node)


# =========================================================
# 4. FIND THE GENERATED FINDING
# =========================================================

finding_id = next(
    node_id
    for node_id, data in kg.graph.nodes(data=True)
    if data.get("type") == "finding"
)


print("\nFINDING ID:")
print(finding_id)


# =========================================================
# 5. FACT CHECK THE FINDING
# =========================================================

verification = fact_checker.verify_finding(
    finding_id
)


print("\n========================================")
print("FACT CHECK RESULT")
print("========================================")

print(verification)


# =========================================================
# 6. EVALUATE EXECUTION
# =========================================================

evaluation = evolution.evaluate_execution(
    result,
    verification
)


print("\n========================================")
print("EXECUTION EVALUATION")
print("========================================")

print(evaluation)


# =========================================================
# 7. STORE EXECUTION HISTORY
# =========================================================

execution_id = evolution.update_knowledge(
    result,
    evaluation
)


print("\n========================================")
print("EXECUTION STORED")
print("========================================")

print(execution_id)

print(
    kg.get_node(execution_id)
)


# =========================================================
# 8. GET SUCCESSFUL EXECUTIONS
# =========================================================

successful = evolution.get_successful_executions()


print("\n========================================")
print("SUCCESSFUL EXECUTIONS")
print("========================================")

print(successful)


# =========================================================
# 9. RECOMMEND AGENT FOR FUTURE TASK
# =========================================================

recommendation = evolution.recommend_workflow(
    "correlation_analysis"
)


print("\n========================================")
print("WORKFLOW RECOMMENDATION")
print("========================================")

print(recommendation)


print("\nRECOMMENDED AGENT:")

print(
    evolution.get_recommended_agent(
        "correlation_analysis"
    )
)


# =========================================================
# 10. FINAL GRAPH
# =========================================================

print("\n========================================")
print("FINAL GRAPH RELATIONSHIPS")
print("========================================")

for edge in kg.graph.edges(data=True):
    print(edge)
    
# Automated integration checks

assert verification["verified"] is True
assert evaluation["successful"] is True

assert execution_id in kg.graph.nodes

assert kg.graph.has_edge(execution_id, finding_id)
assert (
    kg.graph.edges[execution_id, finding_id]["relation"]
    == "produced"
)

assert evolution.get_recommended_agent(
    "correlation_analysis"
) == "EDAAgent"

print("\nALL INTEGRATION ASSERTIONS PASSED")