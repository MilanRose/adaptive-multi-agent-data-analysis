
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
    ):
        """Return a consistent response shape for every route."""
        return {
            "user_request": user_request,
            "selected_agent": selected_agent,
            "status": status,
            "message": message,
            "dataset": dataset,
            "task_count": task_count,
            "insights": insights or [],
            "results": results or [],
            "errors": errors or [],
        }

    def process_request(self, user_request):
        if not isinstance(user_request, str) or not user_request.strip():
            raise ValueError("user_request must be a non-empty string.")

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
            )

        pipeline = run_analysis(str(self.dataset_path))
        converted_results = []
        failed_results = []

        for raw_result in pipeline["results"]:
            if "error" in raw_result:
                failed_results.append(raw_result)
                continue

            agent_result = analysis_result_to_agent_result(
                raw_result,
                dataset=self.dataset_path.name,
                agent=selected_agent,
            )

            finding_ids = self.graph_builder.add_agent_result(
                agent_result
            )

            verification_results = [
                self.fact_checker.verify_finding(finding_id)
                for finding_id in finding_ids
            ]

            # Evaluate the completed task using its verification results.
            # The adapter currently produces one finding per task.
            verification_summary = {
                "verified": (
                    bool(verification_results)
                    and all(
                        check.get("verified") is True
                        for check in verification_results
                    )
                ),
                "checks": {
                    "all_findings_verified": (
                        bool(verification_results)
                        and all(
                            check.get("verified") is True
                            for check in verification_results
                        )
                    )
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
            )

            # Persist findings and execution history so a later process
            # can reload them from the configured storage path.
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

        all_verified = (
            bool(converted_results)
            and not failed_results
            and all(
                check.get("verified") is True
                for item in converted_results
                for check in item["verification"]
            )
        )

        if failed_results:
            status = "completed_with_errors"
        elif all_verified:
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
        )


if __name__ == "__main__":
    orchestrator = Orchestrator()
    request = input("What would you like to analyze? ")
    result = orchestrator.process_request(request)

    print("\nRequest:", result["user_request"])
    print("Selected agent:", result["selected_agent"])
    print("Status:", result["status"])
    print("Message:", result["message"])

    if result["status"] == "completed":
        print("Tasks executed:", result["task_count"])
        print("Findings verified:", sum(
            len(item["finding_ids"])
            for item in result["results"]
        ))
