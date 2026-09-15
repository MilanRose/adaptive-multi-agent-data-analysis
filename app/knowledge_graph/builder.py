from app.knowledge_graph.graph import KnowledgeGraph


class KnowledgeGraphBuilder:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def add_agent_result(self, result):

        dataset_id = result["dataset_id"]
        dataset_name = result["dataset"]

        feature_id = result["feature_id"]
        feature_name = result["feature"]

        finding_id = result["finding_id"]
        finding_description = result["finding"]

        confidence = result.get("confidence", 0.0)

        evidence_id = result["evidence_id"]
        evidence_description = result["evidence"]

        # -------------------------
        # Add Dataset
        # -------------------------

        self.kg.add_node(
            dataset_id,
            "dataset",
            name=dataset_name
        )

        # -------------------------
        # Add Feature
        # -------------------------

        self.kg.add_node(
            feature_id,
            "feature",
            name=feature_name
        )

        # -------------------------
        # Add Finding
        # -------------------------

        self.kg.add_node(
            finding_id,
            "finding",
            description=finding_description,
            confidence=confidence
        )

        # -------------------------
        # Add Evidence
        # -------------------------

        self.kg.add_node(
            evidence_id,
            "evidence",
            evidence=evidence_description
        )

        # -------------------------
        # Add Relationships
        # -------------------------

        self.kg.add_relationship(
            dataset_id,
            "contains",
            feature_id
        )

        self.kg.add_relationship(
            feature_id,
            "supports",
            finding_id
        )

        self.kg.add_relationship(
            finding_id,
            "supported_by",
            evidence_id
        )

        # -------------------------
        # Optional ML Model
        # -------------------------

        if "model_id" in result:

            model_id = result["model_id"]
            model_name = result["model"]

            accuracy = result.get("accuracy")

            self.kg.add_node(
                model_id,
                "model",
                model_name=model_name,
                accuracy=accuracy
            )

            self.kg.add_relationship(
                finding_id,
                "generated_by",
                model_id
            )