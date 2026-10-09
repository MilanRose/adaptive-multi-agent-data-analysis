class Node:

    def __init__(self, node_id, node_type, **attributes):
        self.node_id = node_id
        self.node_type = node_type
        self.attributes = attributes

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "attributes": self.attributes
        }


class DatasetNode(Node):

    def __init__(self, node_id, name, **attributes):
        super().__init__(
            node_id,
            "dataset",
            name=name,
            **attributes
        )


class FeatureNode(Node):

    def __init__(self, node_id, name, **attributes):
        super().__init__(
            node_id,
            "feature",
            name=name,
            **attributes
        )


class FindingNode(Node):

    def __init__(self, node_id, description, **attributes):
        super().__init__(
            node_id,
            "finding",
            description=description,
            **attributes
        )


class EvidenceNode(Node):

    def __init__(self, node_id, evidence, **attributes):
        super().__init__(
            node_id,
            "evidence",
            evidence=evidence,
            **attributes
        )


class ModelNode(Node):

    def __init__(self, node_id, model_name, **attributes):
        super().__init__(
            node_id,
            "model",
            model_name=model_name,
            **attributes
        )