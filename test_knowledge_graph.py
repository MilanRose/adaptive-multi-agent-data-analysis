from app.knowledge_graph.graph import KnowledgeGraph


kg = KnowledgeGraph()

# Add dataset
kg.add_node(
    "customer_churn",
    "dataset",
    filename="customer_churn.csv"
)

# Add features
kg.add_node(
    "monthly_charges",
    "feature"
)

kg.add_node(
    "tenure",
    "feature"
)

# Add finding
kg.add_node(
    "churn_finding_1",
    "finding",
    description="MonthlyCharges is associated with churn"
)

# Add relationships
kg.add_relationship(
    "customer_churn",
    "contains",
    "monthly_charges"
)

kg.add_relationship(
    "customer_churn",
    "contains",
    "tenure"
)

kg.add_relationship(
    "monthly_charges",
    "supports",
    "churn_finding_1"
)


print("Nodes:")
for node in kg.graph.nodes(data=True):
    print(node)

print("\nRelationships:")
for edge in kg.graph.edges(data=True):
    print(edge)