
from pathlib import Path
from tempfile import TemporaryDirectory

from app.knowledge_graph.graph import KnowledgeGraph


with TemporaryDirectory() as temp_dir:
    storage_path = Path(temp_dir) / "knowledge_graph.json"

    # Create a graph with two different relationships
    kg = KnowledgeGraph(storage_path)

    kg.add_node("dataset_1", "dataset", name="customer_churn.csv")
    kg.add_node("finding_1", "finding", confidence=0.84)

    kg.add_relationship("dataset_1", "contains_finding", "finding_1")
    kg.add_relationship("dataset_1", "supports", "finding_1")

    kg.save()

    # Load the graph into a fresh instance
    loaded_kg = KnowledgeGraph(storage_path)

    assert "dataset_1" in loaded_kg.graph.nodes
    assert "finding_1" in loaded_kg.graph.nodes

    relationships = loaded_kg.graph.get_edge_data(
        "dataset_1",
        "finding_1"
    )

    assert relationships is not None
    assert len(relationships) == 2

    relation_types = {
        data["relation"] for data in relationships.values()
    }

    assert relation_types == {"contains_finding", "supports"}

    print("GRAPH PERSISTENCE TEST PASSED")
