from app.schemas import AgentResult


class KnowledgeGraphBuilder:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def add_agent_result(self, result: AgentResult):
        # Generate a unique result number
        result_number = 1

        while any(
            node_id.startswith(f"finding_{result_number}_")
            for node_id in self.kg.graph.nodes
        ):
            result_number += 1

        created_finding_ids = []

        # Create or reuse the dataset node
        dataset_id = f"dataset_{result.dataset}"

        self.kg.add_node(
            dataset_id,
            "dataset",
            name=result.dataset
        )

        # Create finding nodes
        for finding_index, finding_description in enumerate(
            result.findings, start=1
        ):
            finding_id = f"finding_{result_number}_{finding_index}"

            self.kg.add_node(
                finding_id,
                "finding",
                description=finding_description,
                confidence=result.confidence,
                task=result.task,
                agent=result.agent
            )

            created_finding_ids.append(finding_id)

            # Connect dataset to finding
            self.kg.add_relationship(
                dataset_id,
                "contains_finding",
                finding_id
            )

            # Connect explicitly supplied features to matching findings
            for feature_name in result.features:
                if feature_name in finding_description:
                    feature_id = (
                        f"feature_{result.dataset}_{feature_name}"
                    )

                    self.kg.add_node(
                        feature_id,
                        "feature",
                        name=feature_name
                    )

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

            # Create evidence nodes
            for evidence_index, evidence_item in enumerate(
                result.evidence, start=1
            ):
                evidence_id = (
                    f"evidence_{result_number}_"
                    f"{finding_index}_{evidence_index}"
                )

                self.kg.add_node(
                    evidence_id,
                    "evidence",
                    description=evidence_item.description,
                    method=evidence_item.method,
                    value=evidence_item.value
                )

                self.kg.add_relationship(
                    finding_id,
                    "supported_by",
                    evidence_id
                )

            # Create a model node when a model is available
            if result.model is not None:
                model_id = (
                    f"model_{result_number}_{finding_index}"
                )

                self.kg.add_node(
                    model_id,
                    "model",
                    model_name=result.model.name,
                    accuracy=result.model.metrics.get("accuracy"),
                    metrics=result.model.metrics
                )

                self.kg.add_relationship(
                    finding_id,
                    "generated_by",
                    model_id
                )

        return created_finding_ids