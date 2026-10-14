
import inspect
import json
import os
import tempfile
import threading
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


# Shared lock for graph operations across application components.
GRAPH_LOCK = threading.RLock()

# Serializes file replacement operations, including concurrent saves.
_SAVE_LOCK = threading.RLock()


def _node_link_kwargs(function):
    parameters = inspect.signature(function).parameters

    if "edges" in parameters:
        return {"edges": "edges"}

    if "link" in parameters:
        return {"link": "edges"}

    return {}


class KnowledgeGraph:

    def __init__(self, storage_path=None):
        self.graph = nx.MultiDiGraph()
        self.storage_path = (
            Path(storage_path) if storage_path is not None else None
        )

        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_node(self, node_id, node_type, **attributes):
        with GRAPH_LOCK:
            self.graph.add_node(
                node_id,
                type=node_type,
                **attributes
            )

    def add_node_object(self, node):
        with GRAPH_LOCK:
            self.graph.add_node(
                node.node_id,
                type=node.node_type,
                **node.attributes
            )

    def add_relationship(self, source, relation, target):
        with GRAPH_LOCK:
            self.graph.add_edge(
                source,
                target,
                key=relation,
                relation=relation
            )

    def get_node(self, node_id):
        with GRAPH_LOCK:
            return self.graph.nodes[node_id]

    def get_relationships(self, node_id):
        with GRAPH_LOCK:
            return list(
                self.graph.out_edges(
                    node_id,
                    keys=True,
                    data=True
                )
            )

    def get_graph(self):
        # Returns the underlying mutable graph for compatibility.
        # Callers that mutate it directly must acquire GRAPH_LOCK.
        return self.graph


    def save(self, path=None):
        destination = Path(path) if path is not None else self.storage_path

        if destination is None:
            raise ValueError("A storage path must be provided.")

        destination.parent.mkdir(parents=True, exist_ok=True)

        # Serialize saves targeting the same file so concurrent
        # replacements cannot race on Windows.
        with _SAVE_LOCK:
            # Snapshot the graph while graph mutations are locked.
            with GRAPH_LOCK:
                data = json_graph.node_link_data(
                    self.graph,
                    **_node_link_kwargs(json_graph.node_link_data)
                )

            temporary_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)

                    json.dump(data, temporary_file, indent=2)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())

                os.replace(temporary_path, destination)
                temporary_path = None

            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass


    def load(self, path=None):
        source = Path(path) if path is not None else self.storage_path

        if source is None:
            raise ValueError("A storage path must be provided.")

        if not source.exists():
            return

        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)

        loaded_graph = json_graph.node_link_graph(
            data,
            directed=True,
            multigraph=True,
            **_node_link_kwargs(json_graph.node_link_graph)
        )

        with GRAPH_LOCK:
            self.graph = loaded_graph
