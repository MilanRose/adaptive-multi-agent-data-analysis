class GraphReasoner:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def get_supporting_evidence(self, finding_id):

        evidence = []

        for source, target, data in self.kg.graph.edges(
            finding_id,
            data=True
        ):

            if data.get("relation") == "supported_by":

                evidence_data = self.kg.get_node(target)

                evidence.append({
                    "evidence_id": target,
                    "details": evidence_data
                })

        return evidence

    def get_related_nodes(self, node_id):

        relationships = []

        for source, target, data in self.kg.graph.edges(
            node_id,
            data=True
        ):

            relationships.append({
                "source": source,
                "relation": data.get("relation"),
                "target": target
            })

        return relationships

    def get_finding_context(self, finding_id):

        context = {
            "finding": finding_id,
            "features": [],
            "datasets": []
        }

        # Find features that support the finding
        for source, target, data in self.kg.graph.in_edges(
            finding_id,
            data=True
        ):

            if data.get("relation") == "supports":

                context["features"].append(source)

                # Find datasets containing those features
                for dataset_source, feature_target, dataset_data in self.kg.graph.in_edges(
                    source,
                    data=True
                ):

                    if dataset_data.get("relation") == "contains":

                        context["datasets"].append(dataset_source)

        return context
    
    def get_provenance_chain(self, finding_id):

        finding_data = self.kg.get_node(finding_id)

        chain = {
            "finding": {
                "id": finding_id,
                "details": finding_data
            },
            "features": [],
            "datasets": [],
            "evidence": [],
            "models": []
        }

        # Find incoming relationships to the finding
        for source, target, data in self.kg.graph.in_edges(
            finding_id,
            data=True
        ):

            relation = data.get("relation")

            # Feature → Finding
            if relation == "supports":

                feature_data = self.kg.get_node(source)

                chain["features"].append({
                    "id": source,
                    "details": feature_data
                })

                # Dataset → Feature
                for dataset_source, feature_target, dataset_data in self.kg.graph.in_edges(
                    source,
                    data=True
                ):

                    if dataset_data.get("relation") == "contains":

                        dataset_data_value = self.kg.get_node(
                            dataset_source
                        )

                        chain["datasets"].append({
                            "id": dataset_source,
                            "details": dataset_data_value
                        })

        # Find outgoing relationships from the finding
        for source, target, data in self.kg.graph.out_edges(
            finding_id,
            data=True
        ):

            relation = data.get("relation")

            # Finding → Evidence
            if relation == "supported_by":

                evidence_data = self.kg.get_node(target)

                chain["evidence"].append({
                    "id": target,
                    "details": evidence_data
                })

            # Finding → Model
            elif relation == "generated_by":

                model_data = self.kg.get_node(target)

                chain["models"].append({
                    "id": target,
                    "details": model_data
                })

        return chain