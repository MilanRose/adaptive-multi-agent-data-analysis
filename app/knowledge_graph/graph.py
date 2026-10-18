import copy
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

# Serializes file replacement operations within this process.
_SAVE_LOCK = threading.RLock()


def _node_link_kwargs(function):
    """Handle NetworkX node-link API differences across versions."""
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
            Path(storage_path)
            if storage_path is not None
            else None
        )

        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_node(self, node_id, node_type, **attributes):
        """Add or update a node while holding the graph lock."""
        with GRAPH_LOCK:
            self.graph.add_node(
                node_id,
                type=node_type,
                **attributes,
            )

    def add_node_object(self, node):
        """Add a node object while holding the graph lock."""
        with GRAPH_LOCK:
            self.graph.add_node(
                node.node_id,
                type=node.node_type,
                **node.attributes,
            )

    def add_relationship(self, source, relation, target):
        """Add a directed relationship between two nodes."""
        with GRAPH_LOCK:
            self.graph.add_edge(
                source,
                target,
                key=relation,
                relation=relation,
            )

    def get_node(self, node_id):
        """
        Return an independent copy of a node's attributes.

        Modifying the returned dictionary will not modify the
        graph's stored node attributes.
        """
        with GRAPH_LOCK:
            return copy.deepcopy(self.graph.nodes[node_id])

    def get_relationships(self, node_id):
        """
        Return independent copies of a node's outgoing relationships.
        """
        with GRAPH_LOCK:
            relationships = list(
                self.graph.out_edges(
                    node_id,
                    keys=True,
                    data=True,
                )
            )
            return copy.deepcopy(relationships)

    def get_graph(self):
        """
        Return an isolated snapshot with read-only graph topology.

        The snapshot can be inspected without changing the original
        graph's nodes or edges.
        """
        with GRAPH_LOCK:
            snapshot = copy.deepcopy(self.graph)
            nx.freeze(snapshot)
            return snapshot

    def save(self, path=None):
        """
        Atomically save a consistent graph snapshot to JSON.

        The save lock coordinates file replacement between threads
        in this Python process. It does not coordinate other processes.
        """
        destination = (
            Path(path)
            if path is not None
            else self.storage_path
        )

        if destination is None:
            raise ValueError("A storage path must be provided.")

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with _SAVE_LOCK:
            # Snapshot the graph while mutations are locked.
            with GRAPH_LOCK:
                data = json_graph.node_link_data(
                    self.graph,
                    **_node_link_kwargs(json_graph.node_link_data),
                )

            temporary_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)

                    json.dump(
                        data,
                        temporary_file,
                        indent=2,
                    )
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())

                os.replace(
                    temporary_path,
                    destination,
                )
                temporary_path = None

            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def load(self, path=None):
        """Load graph data from a JSON file."""
        source = (
            Path(path)
            if path is not None
            else self.storage_path
        )

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
            **_node_link_kwargs(json_graph.node_link_graph),
        )

        with GRAPH_LOCK:
            self.graph = loaded_graph