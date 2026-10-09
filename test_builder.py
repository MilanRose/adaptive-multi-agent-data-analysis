from app.schemas import AgentResult, Evidence, ModelResult
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder


# ---------------------------------------------------------
# Create Knowledge Graph
# ---------------------------------------------------------

kg = KnowledgeGraph()

# Create Builder
builder = KnowledgeGraphBuilder(kg)


# ---------------------------------------------------------
# Result 1 - EDA Agent
# ---------------------------------------------------------

eda_result = AgentResult(
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


# ---------------------------------------------------------
# Result 2 - ML Agent
# ---------------------------------------------------------

ml_result = AgentResult(
    agent="MLAgent",
    dataset="customer_churn.csv",
    task="classification",

    findings=[
        "Random Forest predicts customer churn"
    ],

    evidence=[
        Evidence(
            description="Random Forest accuracy = 0.87",
            method="Model evaluation",
            value=0.87
        )
    ],

    model=ModelResult(
        name="Random Forest",
        metrics={
            "accuracy": 0.87
        }
    ),

    confidence=0.81
)


# ---------------------------------------------------------
# Add BOTH results to the SAME Knowledge Graph
# ---------------------------------------------------------

builder.add_agent_result(eda_result)
builder.add_agent_result(ml_result)


# ---------------------------------------------------------
# Display nodes
# ---------------------------------------------------------

print("\nNODES:")

for node in kg.graph.nodes(data=True):
    print(node)


# ---------------------------------------------------------
# Display relationships
# ---------------------------------------------------------

print("\nRELATIONSHIPS:")

for edge in kg.graph.edges(data=True):
    print(edge)