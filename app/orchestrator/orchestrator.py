import uuid
from pathlib import Path

from app.orchestrator.router import AdaptiveRouter
from app.analysis_adapter import analysis_result_to_agent_result
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.knowledge_graph.graph import KnowledgeGraph
from app.verification.fact_checker import FactChecker
from src.analysis import run_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Orchestrator:
    """Route requests and execute the integrated EDA workflow."""

    def __init__(
        self,
        dataset_path="data/titanic.csv",
        graph_path=None,
    ):
        self.router = AdaptiveRouter()

        dataset = Path(dataset_path)
        if not dataset.is_absolute():
            dataset = PROJECT_ROOT / dataset
        self.dataset_path = dataset

        if graph_path is None:
            graph_path = PROJECT_ROOT / "outputs" / "knowledge_graph.json"

        self.knowledge_graph = KnowledgeGraph(
            storage_path=graph_path
        )
        self.graph_builder = KnowledgeGraphBuilder(
            self.knowledge_graph
        )
        self.fact_checker = FactChecker(self.knowledge_graph)
        self.evolution_engine = SelfEvolutionEngine(
            self.knowledge_graph
        )

    def _response(
        self,
        user_request,
        selected_agent,
        status,
        *,
        message="",
        dataset=None,
        task_count=0,
        insights=None,
        results=None,
        errors=None,
        run_id=None,
    ):
        """Return a consistent response shape for every route."""
        return {
            "user_request": user_request,
            "run_id": run_id,
            "selected_agent": selected_agent,
            "status": status,
            "message": message,
            "dataset": dataset,
            "task_count": task_count,
            "insights": insights or [],
            "results": results or [],
            "errors": errors or [],
        }

    def process_request(self, user_request, run_id=None):
        """
        Process a user request.

        Supply the same run_id when retrying the same request.
        A new run_id represents a new execution.
        """
        if (
            not isinstance(user_request, str)
            or not user_request.strip()
        ):
            raise ValueError(
                "user_request must be a non-empty string."
            )

        # Generate an ID for a new run, or validate a retry ID.
        if run_id is None:
            run_id = uuid.uuid4().hex
        elif not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )
        else:
            run_id = run_id.strip()

        selected_agent = self.router.route(user_request)

        # Routing exists for these agents, but their execution pipelines
        # have not yet been integrated.
        if selected_agent != "EDAAgent":
            return self._response(
                user_request,
                selected_agent,
                "not_implemented",
                message=(
                    f"{selected_agent} has been selected, but its execution "
                    "workflow is not yet connected."
                ),
                run_id=run_id,
            )

        pipeline = run_analysis(str(self.dataset_path))
        converted_results = []
        failed_results = []

        for result_index, raw_result in enumerate(pipeline["results"]):
            if "error" in raw_result:
                failed_results.append(raw_result)
                continue

            agent_result = analysis_result_to_agent_result(
                raw_result,
                dataset=self.dataset_path.name,
                agent=selected_agent,
            )

            # Use the task ID to derive a stable ID for this task's run.
            # Fall back to the task's position if it has no task ID.
            raw_task_id = raw_result.get("task_id")
            if raw_task_id is None:
                raw_task_id = result_index

            task_run_id = f"{run_id}:{raw_task_id}"

            finding_ids = self.graph_builder.add_agent_result(
                agent_result,
                run_id=task_run_id,
            )

            verification_results = [
                self.fact_checker.verify_finding(finding_id)
                for finding_id in finding_ids
            ]

            # Assess structural validity without claiming scientific proof.
            structurally_verified = (
                bool(verification_results)
                and all(
                    check.get(
                        "structural_verified",
                        check.get("verified", False),
                    ) is True
                    for check in verification_results
                )
            )

            verification_summary = {
                "verified": False,
                "structural_verified": structurally_verified,
                "verification_scope": "structural_only",
                "scientific_support_assessed": False,
                "checks": {
                    "all_findings_structurally_verified": (
                        structurally_verified
                    ),
                },
            }

            evaluation = self.evolution_engine.evaluate_execution(
                agent_result,
                verification_summary,
            )

            self.evolution_engine.update_knowledge(
                agent_result,
                evaluation,
                finding_ids,
                run_id=task_run_id,
            )

            # Persist findings and execution history.
            self.knowledge_graph.save()

            converted_results.append({
                "task_id": raw_result.get("task_id"),
                "task": raw_result.get("task"),
                "analysis_result": raw_result,
                "findings": agent_result.findings,
                "finding_ids": finding_ids,
                "verification": verification_results,
                "evaluation": evaluation,
            })

        # Determine whether the workflow completed with structurally
        # valid findings. This does not establish scientific truth.
        all_structurally_verified = (
            bool(converted_results)
            and not failed_results
            and all(
                check.get(
                    "structural_verified",
                    check.get("verified", False),
                ) is True
                for item in converted_results
                for check in item["verification"]
            )
        )

        if failed_results:
            status = "completed_with_errors"
        elif all_structurally_verified:
            status = "completed"
        else:
            status = "completed_with_warnings"

        return self._response(
            user_request,
            selected_agent,
            status,
            message="EDA workflow executed.",
            dataset=pipeline["dataset"],
            task_count=len(pipeline["tasks"]),
            insights=pipeline["insights"],
            results=converted_results,
            errors=failed_results,
            run_id=run_id,
        )


if __name__ == "__main__":
    orchestrator = Orchestrator()
    request = input("What would you like to analyze? ")
    result = orchestrator.process_request(request)

    print("\nRequest:", result["user_request"])
    print("Run ID:", result["run_id"])
    print("Selected agent:", result["selected_agent"])
    print("Status:", result["status"])
    print("Message:", result["message"])

    if result["status"] == "completed":
        print("Tasks executed:", result["task_count"])
        print(
            "Findings structurally validated:",
            sum(
                len(item["finding_ids"])
                for item in result["results"]
            ),
        )