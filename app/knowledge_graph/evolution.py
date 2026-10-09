class SelfEvolutionEngine:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    # ---------------------------------------------------------
    # 1. EVALUATE AN EXECUTION
    # ---------------------------------------------------------

    def evaluate_execution(self, result, verification):

        checks = verification.get("checks", {})

        verification_passed = verification.get(
            "verified",
            False
        )

        confidence = result.confidence

        model_performance = None

        if result.model:
            metrics = result.model.metrics

            if "accuracy" in metrics:
                model_performance = metrics["accuracy"]

        successful = (
            verification_passed
            and confidence >= 0.7
        )

        if model_performance is not None:
            successful = (
                successful
                and model_performance >= 0.7
            )

        return {
            "successful": successful,
            "confidence": confidence,
            "model_performance": model_performance,
            "verification_passed": verification_passed,
            "checks": checks
        }

    # ---------------------------------------------------------
    # 2. STORE EXECUTION HISTORY
    # ---------------------------------------------------------


    def update_knowledge(self, result, evaluation):

        # Generate a unique execution ID
        execution_number = 1

        while f"execution_{execution_number}" in self.kg.graph.nodes:
            execution_number += 1

        execution_id = f"execution_{execution_number}"

        # Store execution history
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

        # Connect the execution to findings from this agent and task.
        # Match the dataset too, so unrelated results are not linked.
        for node_id, data in list(self.kg.graph.nodes(data=True)):

            if data.get("type") != "finding":
                continue

            if data.get("agent") != result.agent:
                continue

            if data.get("task") != result.task:
                continue

            if not any(
                node_data.get("type") == "dataset"
                and node_data.get("name") == result.dataset
                and self.kg.graph.has_edge(node_id, dataset_id)
                and self.kg.graph.edges[node_id, dataset_id].get("relation")
                    == "belongs_to"
                for dataset_id, node_data in self.kg.graph.nodes(data=True)
            ):
                # Check the dataset through the existing
                # dataset -> finding relationship instead.
                matching_dataset = any(
                    node_data.get("type") == "dataset"
                    and node_data.get("name") == result.dataset
                    and self.kg.graph.has_edge(dataset_id, node_id)
                    and self.kg.graph.edges[dataset_id, node_id].get("relation")
                        == "contains_finding"
                    for dataset_id, node_data in self.kg.graph.nodes(data=True)
                )

                if not matching_dataset:
                    continue

            self.kg.add_relationship(
                execution_id,
                "produced",
                node_id
            )

        return execution_id

    # ---------------------------------------------------------
    # 3. GET SUCCESSFUL EXECUTIONS
    # ---------------------------------------------------------

    def get_successful_executions(self):

        successful_executions = []

        for node_id, data in self.kg.graph.nodes(data=True):

            if data.get("type") == "execution":

                if data.get("successful") is True:

                    successful_executions.append({
                        "execution_id": node_id,
                        "agent": data.get("agent"),
                        "task": data.get("task"),
                        "dataset": data.get("dataset"),
                        "confidence": data.get("confidence"),
                        "model_performance": data.get(
                            "model_performance"
                        )
                    })

        return successful_executions

    # ---------------------------------------------------------
    # 4. GET FAILED EXECUTIONS
    # ---------------------------------------------------------

    def get_failed_executions(self):

        failed_executions = []

        for node_id, data in self.kg.graph.nodes(data=True):

            if data.get("type") == "execution":

                if data.get("successful") is False:

                    failed_executions.append({
                        "execution_id": node_id,
                        "agent": data.get("agent"),
                        "task": data.get("task"),
                        "dataset": data.get("dataset"),
                        "confidence": data.get("confidence"),
                        "model_performance": data.get(
                            "model_performance"
                        )
                    })

        return failed_executions

    # ---------------------------------------------------------
    # 5. FIND SIMILAR EXECUTIONS
    # ---------------------------------------------------------

    def find_similar_executions(self, task):

        similar_executions = []

        for node_id, data in self.kg.graph.nodes(data=True):

            if data.get("type") == "execution":

                if data.get("task") == task:

                    similar_executions.append({
                        "execution_id": node_id,
                        "agent": data.get("agent"),
                        "task": data.get("task"),
                        "dataset": data.get("dataset"),
                        "confidence": data.get("confidence"),
                        "model_performance": data.get(
                            "model_performance"
                        ),
                        "successful": data.get("successful")
                    })

        return similar_executions

    # ---------------------------------------------------------
    # 6. RECOMMEND WORKFLOW
    # ---------------------------------------------------------

    def recommend_workflow(self, task):

        similar_executions = self.find_similar_executions(task)

        if not similar_executions:

            return {
                "recommended_agent": None,
                "task": task,
                "reason": "No previous execution found."
            }

        successful_executions = [
            execution
            for execution in similar_executions
            if execution["successful"] is True
        ]

        failed_executions = [
            execution
            for execution in similar_executions
            if execution["successful"] is False
        ]

        if not successful_executions:

            return {
                "recommended_agent": None,
                "task": task,
                "reason": (
                    "No successful previous execution found."
                ),
                "failed_agents": [
                    execution["agent"]
                    for execution in failed_executions
                ]
            }

        # Agents that have failed this task
        failed_agents = {
            execution["agent"]
            for execution in failed_executions
        }

        # Successful agents that have NOT failed this task
        available_successful = [
            execution
            for execution in successful_executions
            if execution["agent"] not in failed_agents
        ]

        # If every successful agent has also failed
        if not available_successful:

            return {
                "recommended_agent": None,
                "task": task,
                "reason": (
                    "All previously successful agents "
                    "also have failed executions for this task."
                ),
                "failed_agents": list(failed_agents)
            }

        # Select the best performing successful agent
        best_execution = max(
            available_successful,
            key=lambda execution: (
                execution["model_performance"]
                if execution["model_performance"] is not None
                else execution["confidence"]
            )
        )

        return {
            "recommended_agent": best_execution["agent"],
            "task": best_execution["task"],
            "reason": (
                "Recommended based on previous successful "
                "executions while avoiding agents that "
                "previously failed this task."
            ),
            "model_performance": best_execution[
                "model_performance"
            ],
            "confidence": best_execution[
                "confidence"
            ],
            "failed_agents": list(failed_agents)
        }

    # ---------------------------------------------------------
    # 7. GET RECOMMENDED AGENT
    # ---------------------------------------------------------

    def get_recommended_agent(self, task):

        recommendation = self.recommend_workflow(task)

        if recommendation is None:
            return None

        return recommendation.get(
            "recommended_agent"
        )

    # ---------------------------------------------------------
    # 8. ADAPTIVE AGENT RECOMMENDATION
    # ---------------------------------------------------------

    def recommend_agent(self, task):

        executions = []

        # Collect previous executions for this task
        for node_id, data in self.kg.graph.nodes(data=True):

            if data.get("type") == "execution":

                if data.get("task") == task:

                    executions.append({
                        "agent": data.get("agent"),
                        "successful": data.get("successful"),
                        "confidence": data.get("confidence"),
                        "model_performance": data.get(
                            "model_performance"
                        )
                    })

        # -----------------------------------------------------
        # No history
        # -----------------------------------------------------

        if not executions:

            return {
                "recommended_agent": None,
                "task": task,
                "reason": (
                    "No previous execution found. "
                    "A new agent selection is required."
                )
            }

        # -----------------------------------------------------
        # Separate successful and failed executions
        # -----------------------------------------------------

        successful = [
            execution
            for execution in executions
            if execution["successful"] is True
        ]

        failed = [
            execution
            for execution in executions
            if execution["successful"] is False
        ]

        failed_agents = {
            execution["agent"]
            for execution in failed
        }

        # -----------------------------------------------------
        # No successful agent
        # -----------------------------------------------------

        if not successful:

            return {
                "recommended_agent": None,
                "task": task,
                "reason": (
                    "No successful previous execution found."
                ),
                "failed_agents": list(failed_agents)
            }

        # -----------------------------------------------------
        # Remove agents that previously failed
        # -----------------------------------------------------

        available_agents = [
            execution
            for execution in successful
            if execution["agent"] not in failed_agents
        ]

        # -----------------------------------------------------
        # All successful agents have also failed
        # -----------------------------------------------------

        if not available_agents:

            return {
                "recommended_agent": None,
                "task": task,
                "reason": (
                    "All previously successful agents "
                    "also have failed executions for this task."
                ),
                "failed_agents": list(failed_agents)
            }

        # -----------------------------------------------------
        # Select best remaining agent
        # -----------------------------------------------------

        best = max(
            available_agents,
            key=lambda execution: (
                execution["model_performance"]
                if execution["model_performance"] is not None
                else execution["confidence"]
            )
        )

        return {
            "recommended_agent": best["agent"],
            "task": task,
            "reason": (
                "Recommended based on previous successful "
                "executions while avoiding agents that "
                "previously failed this task."
            ),
            "confidence": best["confidence"],
            "model_performance": best["model_performance"],
            "failed_agents": list(failed_agents)
        }