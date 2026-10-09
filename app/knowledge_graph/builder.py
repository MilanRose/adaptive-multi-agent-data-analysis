from app.knowledge_graph.graph import KnowledgeGraph
from app.schemas import AgentResult


class KnowledgeGraphBuilder:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def add_agent_result(self, result: AgentResult):

        # ---------------------------------------------------------
        # Create a unique result prefix
        # ---------------------------------------------------------

        result_id = len(
            [
                node_id
                for node_id, data in self.kg.graph.nodes(data=True)
                if data.get("type") == "finding"
            ]
        ) + 1

        # ---------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------

        dataset_id = f"dataset_{result.dataset}"

        self.kg.add_node(
            dataset_id,
            "dataset",
            name=result.dataset
        )

        # ---------------------------------------------------------
        # Findings
        # ---------------------------------------------------------

        for finding_index, finding_description in enumerate(
            result.findings,
            start=1
        ):

            finding_id = (
                f"finding_{result_id}_{finding_index}"
            )

            self.kg.add_node(
                finding_id,
                "finding",
                description=finding_description,
                confidence=result.confidence,
                task=result.task,
                agent=result.agent
            )

            self.kg.add_relationship(
                dataset_id,
                "contains_finding",
                finding_id
            )

            # -----------------------------------------------------
            # Evidence
            # -----------------------------------------------------

            for evidence_index, evidence in enumerate(
                result.evidence,
                start=1
            ):

                evidence_id = (
                    f"evidence_"
                    f"{result_id}_"
                    f"{finding_index}_"
                    f"{evidence_index}"
                )

                self.kg.add_node(
                    evidence_id,
                    "evidence",
                    description=evidence.description,
                    method=evidence.method,
                    value=evidence.value
                )

                self.kg.add_relationship(
                    finding_id,
                    "supported_by",
                    evidence_id
                )

            # -----------------------------------------------------
            # Model
            # -----------------------------------------------------

            if result.model is not None:

                model_id = (
                    f"model_"
                    f"{result_id}_"
                    f"{finding_index}"
                )

                accuracy = result.model.metrics.get(
                    "accuracy"
                )

                self.kg.add_node(
                    model_id,
                    "model",
                    model_name=result.model.name,
                    accuracy=accuracy,
                    metrics=result.model.metrics
                )

                self.kg.add_relationship(
                    finding_id,
                    "generated_by",
                    model_id
                )

        return self.kg