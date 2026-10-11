
import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


class KnowledgeGraph:

    def __init__(self, storage_path=None):
        self.graph = nx.MultiDiGraph()
        self.storage_path = (
            Path(storage_path) if storage_path is not None else None
        )

        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_node(self, node_id, node_type, **attributes):
        self.graph.add_node(
            node_id,
            type=node_type,
            **attributes
        )

    def add_node_object(self, node):
        self.graph.add_node(
            node.node_id,
            type=node.node_type,
            **node.attributes
        )

    def add_relationship(self, source, relation, target):
        self.graph.add_edge(
            source,
            target,
            key=relation,
            relation=relation
        )

    def get_node(self, node_id):
        return self.graph.nodes[node_id]

    def get_relationships(self, node_id):
        return list(
            self.graph.out_edges(
                node_id,
                keys=True,
                data=True
            )
        )

    def get_graph(self):
        return self.graph

    def save(self, path=None):
        destination = Path(path) if path else self.storage_path

        if destination is None:
            raise ValueError("A storage path must be provided.")

        destination.parent.mkdir(parents=True, exist_ok=True)

        data = json_graph.node_link_data(
            self.graph,
            edges="edges"
        )

        with destination.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def load(self, path=None):
        source = Path(path) if path else self.storage_path

        if source is None:
            raise ValueError("A storage path must be provided.")

        if not source.exists():
            return

        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.graph = json_graph.node_link_graph(
            data,
            directed=True,
            multigraph=True,
            edges="edges"
        )
