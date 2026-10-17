from app.orchestrator.router import AdaptiveRouter


class Orchestrator:
    """Coordinate requests by selecting an agent."""

    def __init__(self):
        self.router = AdaptiveRouter()

    def process_request(self, user_request):
        selected_agent = self.router.route(user_request)

        return {
            "user_request": user_request,
            "selected_agent": selected_agent,
            "status": "Agent selected successfully"
        }


if __name__ == "__main__":
    orchestrator = Orchestrator()

    request = input("What would you like to analyze? ")
    result = orchestrator.process_request(request)

    print("\nRequest:", result["user_request"])
    print("Selected agent:", result["selected_agent"])
    print("Status:", result["status"])
