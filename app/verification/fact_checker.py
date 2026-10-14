
import math

from app.knowledge_graph.graph import GRAPH_LOCK


class FactChecker:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def verify_finding(self, finding_id):
        with GRAPH_LOCK:
            # Check whether the node exists.
            if finding_id not in self.kg.graph.nodes:
                return {
                    "finding_id": finding_id,
                    "verified": False,
                    "reason": "Finding does not exist",
                    "confidence": 0.0,
                    "evidence": [],
                    "models": [],
                    "checks": {
                        "finding_exists": False,
                        "is_finding": False,
                        "has_confidence": False,
                        "has_evidence": False,
                        "evidence_connected": False,
                        "has_model": False,
                        "valid_confidence": False,
                        "valid_model_accuracy": False
                    }
                }

            finding = dict(self.kg.get_node(finding_id))

            # Reject nodes that are not findings.
            is_finding = finding.get("type") == "finding"

            if not is_finding:
                return {
                    "finding_id": finding_id,
                    "verified": False,
                    "reason": "Node is not a finding",
                    "confidence": 0.0,
                    "evidence": [],
                    "models": [],
                    "checks": {
                        "finding_exists": True,
                        "is_finding": False,
                        "has_confidence": False,
                        "has_evidence": False,
                        "evidence_connected": False,
                        "has_model": False,
                        "valid_confidence": False,
                        "valid_model_accuracy": False
                    }
                }

            # Collect evidence and model relationships.
            evidence_ids = []
            model_ids = []
            invalid_evidence_targets = []
            invalid_model_targets = []

            for _, target, data in self.kg.graph.out_edges(
                finding_id, data=True
            ):
                relation = data.get("relation")

                if relation == "supported_by":
                    if (
                        target in self.kg.graph.nodes
                        and self.kg.get_node(target).get("type") == "evidence"
                    ):
                        evidence_ids.append(target)
                    else:
                        invalid_evidence_targets.append(target)

                elif relation == "generated_by":
                    if (
                        target in self.kg.graph.nodes
                        and self.kg.get_node(target).get("type") == "model"
                    ):
                        model_ids.append(target)
                    else:
                        invalid_model_targets.append(target)

            # Validate finding confidence.
            confidence = finding.get("confidence")
            has_confidence = confidence is not None

            valid_confidence = (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and 0.0 <= confidence <= 1.0
                and math.isfinite(confidence)
            )

            # Validate evidence relationships.
            has_evidence = bool(evidence_ids)

            evidence_connected = (
                has_evidence
                and not invalid_evidence_targets
            )

            # Models are optional, but connected models must be valid.
            has_model = bool(model_ids)
            valid_model_accuracy = not invalid_model_targets

            for model_id in model_ids:
                model = self.kg.get_node(model_id)
                accuracy = model.get("accuracy")

                # Missing accuracy is allowed.
                # If present, it must be numeric and within [0, 1].
                if accuracy is not None and (
                    not isinstance(accuracy, (int, float))
                    or isinstance(accuracy, bool)
                    or not 0.0 <= accuracy <= 1.0
                    or not math.isfinite(accuracy)
                ):
                    valid_model_accuracy = False
                    break

            checks = {
                "finding_exists": True,
                "is_finding": True,
                "has_confidence": has_confidence,
                "has_evidence": has_evidence,
                "evidence_connected": evidence_connected,
                "has_model": has_model,
                "valid_confidence": valid_confidence,
                "valid_model_accuracy": valid_model_accuracy
            }

            # Confidence and valid evidence are required.
            # A model is optional.
            verified = all([
                checks["finding_exists"],
                checks["is_finding"],
                checks["has_confidence"],
                checks["has_evidence"],
                checks["evidence_connected"],
                checks["valid_confidence"],
                checks["valid_model_accuracy"]
            ])

            return {
                "finding_id": finding_id,
                "verified": verified,
                "confidence": confidence if has_confidence else 0.0,
                "evidence": evidence_ids,
                "models": model_ids,
                "checks": checks
            }
