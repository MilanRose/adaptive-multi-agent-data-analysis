import hashlib
import re

from app.knowledge_graph.graph import GRAPH_LOCK
from app.schemas import AgentResult


class KnowledgeGraphBuilder:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def add_agent_result(self, result: AgentResult):
        with GRAPH_LOCK:
            return self._add_agent_result(result)

    def _add_agent_result(self, result: AgentResult):
        # Validate evidence references before modifying the graph.
        for evidence_item in result.evidence:
            for finding_index in evidence_item.finding_indices:
                if not 0 <= finding_index < len(result.findings):
                    raise ValueError(
                        f"Evidence '{evidence_item.description}' refers "
                        f"to invalid finding index {finding_index}. "
                        f"Valid indices are 0 to "
                        f"{len(result.findings) - 1}."
                    )

        # Find a result number that does not collide with existing findings.
        result_number = 1

        while any(
            node_id.startswith(f"finding_{result_number}_")
            for node_id in self.kg.graph.nodes
        ):
            result_number += 1

        created_finding_ids = []

        # Create or reuse the dataset node.
        dataset_id = f"dataset_{result.dataset}"

        self.kg.add_node(
            dataset_id,
            "dataset",
            name=result.dataset,
        )

        # Create finding nodes.
        for finding_index, finding_description in enumerate(
            result.findings
        ):
            finding_id = f"finding_{result_number}_{finding_index + 1}"

            self.kg.add_node(
                finding_id,
                "finding",
                description=finding_description,
                confidence=result.confidence,
                task=result.task,
                agent=result.agent,
            )

            created_finding_ids.append(finding_id)

            # Link the dataset directly to the finding.
            self.kg.add_relationship(
                dataset_id,
                "contains_finding",
                finding_id,
            )

            # Match complete, non-empty feature names.
            for raw_feature_name in result.features:
                feature_name = raw_feature_name.strip()

                if not feature_name:
                    continue

                if not re.search(
                    rf"(?<!\w){re.escape(feature_name)}(?!\w)",
                    finding_description,
                    flags=re.IGNORECASE,
                ):
                    continue

                # Preserve the original ID when it is safe to reuse.
                legacy_feature_id = (
                    f"feature_{result.dataset}_{feature_name}"
                )

                existing_data = self.kg.graph.nodes.get(
                    legacy_feature_id, {}
                )

                same_feature = (
                    existing_data.get("type") == "feature"
                    and existing_data.get("dataset") == result.dataset
                    and existing_data.get("name") == feature_name
                )

                if (
                    legacy_feature_id not in self.kg.graph
                    or same_feature
                ):
                    feature_id = legacy_feature_id
                else:
                    # Generate a deterministic ID for a collision.
                    identity = f"{result.dataset}\0{feature_name}"
                    suffix = hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()[:12]

                    feature_id = f"{legacy_feature_id}_{suffix}"

                    # Guard against an existing ID belonging to
                    # another node or feature.
                    while feature_id in self.kg.graph:
                        existing_data = self.kg.graph.nodes[feature_id]

                        if (
                            existing_data.get("type") == "feature"
                            and existing_data.get("dataset")
                            == result.dataset
                            and existing_data.get("name")
                            == feature_name
                        ):
                            break

                        suffix = hashlib.sha256(
                            f"{identity}\0{suffix}".encode("utf-8")
                        ).hexdigest()[:12]

                        feature_id = (
                            f"{legacy_feature_id}_{suffix}"
                        )

                # Create or update the feature node.
                self.kg.add_node(
                    feature_id,
                    "feature",
                    name=feature_name,
                    dataset=result.dataset,
                )

                # Link dataset to feature.
                self.kg.add_relationship(
                    dataset_id,
                    "contains",
                    feature_id,
                )

                # Link feature to its finding.
                self.kg.add_relationship(
                    feature_id,
                    "supports",
                    finding_id,
                )

            # Create a model node when a model is available.
            if result.model is not None:
                model_id = (
                    f"model_{result_number}_{finding_index + 1}"
                )

                self.kg.add_node(
                    model_id,
                    "model",
                    model_name=result.model.name,
                    accuracy=result.model.metrics.get("accuracy"),
                    metrics=result.model.metrics,
                )

                self.kg.add_relationship(
                    finding_id,
                    "generated_by",
                    model_id,
                )

        # Connect evidence only to its specified findings.
        for evidence_index, evidence_item in enumerate(
            result.evidence,
            start=1,
        ):
            evidence_id = (
                f"evidence_{result_number}_{evidence_index}"
            )

            self.kg.add_node(
                evidence_id,
                "evidence",
                description=evidence_item.description,
                method=evidence_item.method,
                value=evidence_item.value,
            )

            for finding_index in evidence_item.finding_indices:
                finding_id = created_finding_ids[finding_index]

                self.kg.add_relationship(
                    finding_id,
                    "supported_by",
                    evidence_id,
                )

        return created_finding_ids