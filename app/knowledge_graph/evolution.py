
import math

from app.knowledge_graph.graph import GRAPH_LOCK


class SelfEvolutionEngine:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def evaluate_execution(self, result, verification):
        checks = verification.get("checks", {})
        verification_passed = verification.get("verified", False)

        confidence = result.confidence

        # Validate confidence before using it.
        valid_confidence = (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        )

        model_performance = None
        if result.model:
            model_performance = result.model.metrics.get("accuracy")

        valid_model_performance = (
            model_performance is None
            or (
                isinstance(model_performance, (int, float))
                and not isinstance(model_performance, bool)
                and math.isfinite(model_performance)
                and 0.0 <= model_performance <= 1.0
            )
        )

        successful = (
            verification_passed is True
            and valid_confidence
            and confidence >= 0.7
            and valid_model_performance
        )

        if model_performance is not None and valid_model_performance:
            successful = successful and model_performance >= 0.7

        return {
            "successful": bool(successful),
            "confidence": confidence,
            "model_performance": model_performance,
            "verification_passed": verification_passed,
            "checks": checks
        }

    def update_knowledge(self, result, evaluation, finding_ids):
        with GRAPH_LOCK:
            execution_number = 1

            while f"execution_{execution_number}" in self.kg.graph.nodes:
                execution_number += 1

            execution_id = f"execution_{execution_number}"

            self.kg.add_node(
                execution_id,
                "execution",
                agent=result.agent,
                task=result.task,
                dataset=result.dataset,
                successful=evaluation["successful"],
                confidence=evaluation["confidence"],
                model_performance=evaluation["model_performance"],
                verification_passed=evaluation["verification_passed"]
            )

            for finding_id in finding_ids:
                if (
                    finding_id in self.kg.graph.nodes
                    and self.kg.graph.nodes[finding_id].get("type")
                    == "finding"
                ):
                    self.kg.add_relationship(
                        execution_id,
                        "produced",
                        finding_id
                    )

            return execution_id

    def _get_executions(self, successful=None, task=None, dataset=None):
        with GRAPH_LOCK:
            executions = []

            for node_id, data in self.kg.graph.nodes(data=True):
                if data.get("type") != "execution":
                    continue

                if task is not None and data.get("task") != task:
                    continue

                if dataset is not None and data.get("dataset") != dataset:
                    continue

                if (
                    successful is not None
                    and data.get("successful") is not successful
                ):
                    continue

                executions.append({
                    "execution_id": node_id,
                    "agent": data.get("agent"),
                    "task": data.get("task"),
                    "dataset": data.get("dataset"),
                    "successful": data.get("successful"),
                    "confidence": data.get("confidence"),
                    "model_performance": data.get("model_performance")
                })

            return executions

    def get_successful_executions(self):
        return self._get_executions(successful=True)

    def get_failed_executions(self):
        return self._get_executions(successful=False)

    def find_similar_executions(self, task, dataset=None):
        return self._get_executions(task=task, dataset=dataset)

    def _recommend(self, task, dataset=None):
        executions = self.find_similar_executions(task)

        if dataset is not None:
            executions = [
                execution for execution in executions
                if execution["dataset"] == dataset
            ]
        else:
            datasets = {
                execution["dataset"]
                for execution in executions
            }

            if len(datasets) > 1:
                return {
                    "recommended_agent": None,
                    "task": task,
                    "dataset": None,
                    "reason": (
                        "Multiple datasets have execution history for this "
                        "task. Specify a dataset to avoid mixing histories."
                    )
                }

            if len(datasets) == 1:
                dataset = next(iter(datasets))

        if not executions:
            return {
                "recommended_agent": None,
                "task": task,
                "dataset": dataset,
                "reason": (
                    "No previous execution found for this task and dataset."
                )
            }

        successful = [
            execution for execution in executions
            if execution["successful"] is True
        ]

        failed = [
            execution for execution in executions
            if execution["successful"] is False
        ]

        failed_agents = {
            execution["agent"] for execution in failed
        }

        available = [
            execution for execution in successful
            if execution["agent"] not in failed_agents
            and isinstance(execution["confidence"], (int, float))
            and not isinstance(execution["confidence"], bool)
            and math.isfinite(execution["confidence"])
        ]

        if not available:
            return {
                "recommended_agent": None,
                "task": task,
                "dataset": dataset,
                "reason": "No eligible successful execution is available.",
                "failed_agents": sorted(failed_agents)
            }

        best = max(
            available,
            key=lambda execution: (
                execution["model_performance"]
                if execution["model_performance"] is not None
                else execution["confidence"]
            )
        )

        return {
            "recommended_agent": best["agent"],
            "task": task,
            "dataset": dataset,
            "reason": (
                "Recommended based on model performance when available, "
                "otherwise confidence, while excluding agents with failed "
                "executions in the selected task and dataset history."
            ),
            "confidence": best["confidence"],
            "model_performance": best["model_performance"],
            "failed_agents": sorted(failed_agents)
        }

    def recommend_workflow(self, task, dataset=None):
        return self._recommend(task, dataset)

    def get_recommended_agent(self, task, dataset=None):
        return self.recommend_workflow(
            task, dataset
        ).get("recommended_agent")

    def recommend_agent(self, task, dataset=None):
        return self._recommend(task, dataset)
