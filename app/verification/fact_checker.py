class FactChecker:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def verify_finding(self, finding_id):

        # Check whether the finding exists
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
                    "has_confidence": False,
                    "has_evidence": False,
                    "evidence_connected": False,
                    "has_model": False,
                    "valid_confidence": False,
                    "valid_model_accuracy": False
                }
            }

        finding = self.kg.get_node(finding_id)

        # Collect supporting evidence and connected models
        evidence_ids = []
        model_ids = []

        for _, target, data in self.kg.graph.out_edges(
            finding_id, data=True
        ):
            relation = data.get("relation")

            if relation == "supported_by":
                evidence_ids.append(target)
            elif relation == "generated_by":
                model_ids.append(target)

        confidence = finding.get("confidence")

        has_confidence = confidence is not None
        valid_confidence = (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0.0 <= confidence <= 1.0
        )

        has_evidence = bool(evidence_ids)

        evidence_connected = all(
            evidence_id in self.kg.graph.nodes
            and self.kg.get_node(evidence_id).get("type") == "evidence"
            for evidence_id in evidence_ids
        )

        has_model = bool(model_ids)
        valid_model_accuracy = True

        for model_id in model_ids:
            model = self.kg.get_node(model_id)
            accuracy = model.get("accuracy")

            if accuracy is not None:
                if (
                    not isinstance(accuracy, (int, float))
                    or isinstance(accuracy, bool)
                    or not 0.0 <= accuracy <= 1.0
                ):
                    valid_model_accuracy = False
                    break

        checks = {
            "finding_exists": True,
            "has_confidence": has_confidence,
            "has_evidence": has_evidence,
            "evidence_connected": evidence_connected,
            "has_model": has_model,
            "valid_confidence": valid_confidence,
            "valid_model_accuracy": valid_model_accuracy
        }

        # A model is optional, but confidence and evidence are required.
        verified = all([
            checks["finding_exists"],
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
