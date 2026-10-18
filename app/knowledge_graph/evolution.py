import hashlib
import json
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
            "checks": checks,
        }

    def update_knowledge(
        self, result, evaluation, finding_ids, run_id=None
    ):
        """
        Store execution history.

        When run_id is supplied, an identical retry reuses the
        existing execution instead of creating another one.
        """
        with GRAPH_LOCK:
            execution_metadata = {}

            if run_id is not None:
                if not isinstance(run_id, str) or not run_id.strip():
                    raise ValueError(
                        "run_id must be a non-empty string."
                    )

                run_id = run_id.strip()

                payload = json.dumps(
                    {
                        "result": result.model_dump(mode="json"),
                        "evaluation": evaluation,
                        "finding_ids": list(finding_ids),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )

                execution_fingerprint = hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest()

                # Check whether this execution has already been recorded.
                for node_id, data in self.kg.graph.nodes(data=True):
                    if (
                        data.get("type") == "execution"
                        and data.get("run_id") == run_id
                    ):
                        if (
                            data.get("execution_fingerprint")
                            != execution_fingerprint
                        ):
                            raise ValueError(
                                "run_id has already been used for a "
                                "different execution."
                            )

                        return node_id

                execution_metadata = {
                    "run_id": run_id,
                    "execution_fingerprint": execution_fingerprint,
                }

            # Allocate an unused execution ID.
            execution_number = 1

            while (
                f"execution_{execution_number}"
                in self.kg.graph.nodes
            ):
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
                verification_passed=evaluation["verification_passed"],
                **execution_metadata,
            )

            # Link the execution only to its own valid findings.
            for finding_id in finding_ids:
                if (
                    finding_id in self.kg.graph.nodes
                    and self.kg.graph.nodes[finding_id].get("type")
                    == "finding"
                ):
                    self.kg.add_relationship(
                        execution_id,
                        "produced",
                        finding_id,
                    )

            return execution_id

    def _get_executions(
        self, successful=None, task=None, dataset=None
    ):
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
                    "model_performance": data.get("model_performance"),
                })

            return executions

    def get_successful_executions(self):
        return self._get_executions(successful=True)

    def get_failed_executions(self):
        return self._get_executions(successful=False)

    def find_similar_executions(self, task, dataset=None):
        return self._get_executions(
            task=task,
            dataset=dataset,
        )

    def _recommend(self, task, dataset=None):
        executions = self.find_similar_executions(task)

        # Keep recommendations isolated to the requested dataset.
        if dataset is not None:
            executions = [
                execution
                for execution in executions
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
                    ),
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
                ),
            }

        # Execution IDs are allocated sequentially by update_knowledge.
        # Sort numerically so the latest execution for each agent wins.
        def execution_order(execution):
            node_id = execution["execution_id"]
            prefix = "execution_"

            if not node_id.startswith(prefix):
                return -1

            suffix = node_id[len(prefix):]
            return int(suffix) if suffix.isdigit() else -1

        executions = sorted(executions, key=execution_order)

        # Keep only each agent's latest outcome for this task and dataset.
        latest_by_agent = {}

        for execution in executions:
            agent = execution["agent"]

            if agent is not None:
                latest_by_agent[agent] = execution

        latest_executions = list(latest_by_agent.values())

        # An agent is currently failed only when its latest execution
        # was not successful. Earlier failures do not block recovery.
        failed_agents = {
            agent
            for agent, execution in latest_by_agent.items()
            if execution["successful"] is not True
        }

        available = [
            execution
            for execution in latest_executions
            if execution["successful"] is True
            and isinstance(execution["confidence"], (int, float))
            and not isinstance(execution["confidence"], bool)
            and math.isfinite(execution["confidence"])
        ]

        if not available:
            return {
                "recommended_agent": None,
                "task": task,
                "dataset": dataset,
                "reason": (
                    "No eligible successful latest execution is available."
                ),
                "failed_agents": sorted(failed_agents),
            }

                # Compare like with like: never rank model accuracy against
        # confidence, since they represent different measurements.
        candidates_with_model_performance = [
            execution
            for execution in available
            if (
                isinstance(
                    execution["model_performance"],
                    (int, float),
                )
                and not isinstance(
                    execution["model_performance"],
                    bool,
                )
                and math.isfinite(execution["model_performance"])
                and 0.0 <= execution["model_performance"] <= 1.0
            )
        ]

        if candidates_with_model_performance:
            best = max(
                candidates_with_model_performance,
                key=lambda execution: execution["model_performance"],
            )
            ranking_metric = "model performance"
        else:
            best = max(
                available,
                key=lambda execution: execution["confidence"],
            )
            ranking_metric = "confidence"

        return {
            "recommended_agent": best["agent"],
            "task": task,
            "dataset": dataset,
                        "reason": (
                f"Recommended using {ranking_metric} among eligible "
                "latest successful executions for the selected task "
                "and dataset."
            ),
            "confidence": best["confidence"],
            "model_performance": best["model_performance"],
            "failed_agents": sorted(failed_agents),
        }

    def recommend_workflow(self, task, dataset=None):
        return self._recommend(task, dataset)

    def get_recommended_agent(self, task, dataset=None):
        return self.recommend_workflow(
            task, dataset
        ).get("recommended_agent")

    def recommend_agent(self, task, dataset=None):
        return self._recommend(task, dataset)