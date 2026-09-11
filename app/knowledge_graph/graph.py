import networkx as nx


class KnowledgeGraph:

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id, node_type, **attributes):
        self.graph.add_node(
            node_id,
            type=node_type,
            **attributes
        )

    def add_relationship(self, source, relation, target):
        self.graph.add_edge(
            source,
            target,
            relation=relation
        )

    def get_node(self, node_id):
        return self.graph.nodes[node_id]

    def get_relationships(self, node_id):
        return list(self.graph.edges(
            node_id,
            data=True
        ))

    def get_graph(self):
        return self.graph