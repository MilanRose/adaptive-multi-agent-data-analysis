
from pathlib import Path

from app.orchestrator.router import AdaptiveRouter
from app.analysis_adapter import analysis_result_to_agent_result
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.graph import KnowledgeGraph
from app.verification.fact_checker import FactChecker
from src.analysis import run_analysis


class Orchestrator:
    """Route requests and execute the currently integrated EDA workflow."""

    def __init__(self, dataset_path="data/titanic.csv"):
        self.router = AdaptiveRouter()
        self.dataset_path = dataset_path

        self.knowledge_graph = KnowledgeGraph()
        self.graph_builder = KnowledgeGraphBuilder(self.knowledge_graph)
        self.fact_checker = FactChecker(self.knowledge_graph)

    def process_request(self, user_request):
        if not isinstance(user_request, str) or not user_request.strip():
            raise ValueError("user_request must be a non-empty string.")

        selected_agent = self.router.route(user_request)

        # Other agents are selected by the router, but their execution
        # modules have not yet been integrated.
        if selected_agent != "EDAAgent":
            return {
                "user_request": user_request,
                "selected_agent": selected_agent,
                "status": "not_implemented",
                "message": (
                    f"{selected_agent} has been selected, but its execution "
                    "workflow is not yet connected."
                ),
                "results": [],
            }

        pipeline = run_analysis(self.dataset_path)
        converted_results = []
        failed_results = []

        for raw_result in pipeline["results"]:
            if "error" in raw_result:
                failed_results.append(raw_result)
                continue

            agent_result = analysis_result_to_agent_result(
                raw_result,
                dataset=Path(self.dataset_path).name,
                agent=selected_agent,
            )

            finding_ids = self.graph_builder.add_agent_result(agent_result)

            verification_results = [
                self.fact_checker.verify_finding(finding_id)
                for finding_id in finding_ids
            ]

            converted_results.append({
                "task_id": raw_result.get("task_id"),
                "task": raw_result.get("task"),
                "analysis_result": raw_result,
                "findings": agent_result.findings,
                "finding_ids": finding_ids,
                "verification": verification_results,
            })

        all_verified = (
            bool(converted_results)
            and all(
                check["verified"]
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

        return {
            "user_request": user_request,
            "selected_agent": selected_agent,
            "status": status,
            "dataset": pipeline["dataset"],
            "task_count": len(pipeline["tasks"]),
            "insights": pipeline["insights"],
            "results": converted_results,
            "errors": failed_results,
        }


if __name__ == "__main__":
    orchestrator = Orchestrator()
    request = input("What would you like to analyze? ")
    result = orchestrator.process_request(request)

    print("\nRequest:", result["user_request"])
    print("Selected agent:", result["selected_agent"])
    print("Status:", result["status"])

    if result["status"] == "completed":
        print("Tasks executed:", result["task_count"])
        print("Findings verified:", sum(
            len(item["finding_ids"]) for item in result["results"]
        ))
