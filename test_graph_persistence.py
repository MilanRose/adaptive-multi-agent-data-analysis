from concurrent.futures import ThreadPoolExecutor

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.knowledge_graph.graph import KnowledgeGraph


def test_graph_persistence():
    with TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "knowledge_graph.json"

        # Create a graph with multiple relationships
        kg = KnowledgeGraph(storage_path)
        kg.add_node(
            "dataset_1",
            "dataset",
            name="customer_churn.csv"
        )
        kg.add_node(
            "finding_1",
            "finding",
            confidence=0.84
        )
        kg.add_relationship(
            "dataset_1",
            "contains_finding",
            "finding_1"
        )
        kg.add_relationship(
            "dataset_1",
            "supports",
            "finding_1"
        )

        # Save and verify the file
        kg.save()

        assert storage_path.exists()
        assert storage_path.stat().st_size > 0

        # Load into a fresh graph and verify persistence
        loaded_kg = KnowledgeGraph(storage_path)

        assert "dataset_1" in loaded_kg.graph.nodes
        assert "finding_1" in loaded_kg.graph.nodes
        assert loaded_kg.get_node("dataset_1")["name"] == (
            "customer_churn.csv"
        )
        assert loaded_kg.get_node("finding_1")["confidence"] == 0.84

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

        # Record the original file before attempting a failed save
        original_contents = storage_path.read_bytes()

        # Add a value that JSON cannot serialize
        kg.graph.nodes["finding_1"]["unsupported_value"] = object()

        with pytest.raises(TypeError):
            kg.save()

        # The original file must remain unchanged
        assert storage_path.read_bytes() == original_contents

        # The previous file must still be valid and loadable
        preserved_kg = KnowledgeGraph(storage_path)

        assert "dataset_1" in preserved_kg.graph.nodes
        assert "finding_1" in preserved_kg.graph.nodes
        assert (
            preserved_kg.get_node("finding_1")["confidence"] == 0.84
        )

        # Confirm the file remains valid JSON
        with storage_path.open("r", encoding="utf-8") as file:
            json.load(file)

        print("GRAPH PERSISTENCE AND FAILED-SAVE TEST PASSED")


def test_concurrent_saves_preserve_valid_graph_file():
    with TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "concurrent_graph.json"
        kg = KnowledgeGraph(storage_path)

        kg.add_node("dataset_1", "dataset", name="data.csv")

        for index in range(20):
            finding_id = f"finding_{index}"
            kg.add_node(
                finding_id,
                "finding",
                confidence=0.8
            )
            kg.add_relationship(
                "dataset_1",
                "contains_finding",
                finding_id
            )

        # Multiple threads save the same graph concurrently.
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(kg.save)
                for _ in range(10)
            ]

            for future in futures:
                future.result()

        # The resulting file must remain valid JSON and loadable.
        with storage_path.open("r", encoding="utf-8") as file:
            json.load(file)

        loaded_kg = KnowledgeGraph(storage_path)

        assert "dataset_1" in loaded_kg.graph.nodes

        for index in range(20):
            finding_id = f"finding_{index}"
            assert finding_id in loaded_kg.graph.nodes
            assert loaded_kg.graph.has_edge(
                "dataset_1",
                finding_id
            )
