class FactChecker:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def verify_finding(self, finding_id):

        # Check whether finding exists
        if finding_id not in self.kg.graph.nodes:
            return {
                "finding_id": finding_id,
                "verified": False,
                "reason": "Finding does not exist"
            }

        finding = self.kg.get_node(finding_id)

        checks = {}

        # -------------------------
        # Check 1: Finding exists
        # -------------------------

        checks["finding_exists"] = True

        # -------------------------
        # Check 2: Confidence exists
        # -------------------------

        checks["has_confidence"] = (
            "confidence" in finding
        )

        # -------------------------
        # Check 3: Supporting evidence
        # -------------------------

        evidence = []

        for source, target, data in self.kg.graph.out_edges(
            finding_id,
            data=True
        ):

            if data.get("relation") == "supported_by":
                evidence.append(target)

        checks["has_evidence"] = len(evidence) > 0

        # -------------------------
        # Check 4: Evidence connected
        # -------------------------

        checks["evidence_connected"] = all(
            evidence_id in self.kg.graph.nodes
            for evidence_id in evidence
        )

        # -------------------------
        # Check 5: Model
        # -------------------------

        models = []

        for source, target, data in self.kg.graph.out_edges(
            finding_id,
            data=True
        ):

            if data.get("relation") == "generated_by":
                models.append(target)

        checks["has_model"] = len(models) > 0
        
        # -------------------------
        # Check 6: Confidence validity
        # -------------------------

        confidence = finding.get("confidence")

        checks["valid_confidence"] = (
            isinstance(confidence, (int, float))
            and 0 <= confidence <= 1
            )

        # -------------------------
        # Check 7: Model validity
        # -------------------------

        valid_model = True

        for model_id in models:
            model_details = self.kg.get_node(model_id)

            accuracy = model_details.get("accuracy")

            if accuracy is not None:

                if not (
                    isinstance(accuracy, (int, float))
                    and 0 <= accuracy <= 1
                ):
                    valid_model = False

        checks["valid_model_accuracy"] = valid_model

        # -------------------------
        # Overall verification
        # -------------------------

        verified = all(checks.values())

        return {
            "finding_id": finding_id,
            "verified": verified,
            "confidence": finding.get("confidence"),
            "evidence": evidence,
            "models": models,
            "checks": checks
        }