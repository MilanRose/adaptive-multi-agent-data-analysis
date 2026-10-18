
class AdaptiveRouter:
    """Select an agent based on the user's request."""

    def route(self, task):
        task = task.lower()

        if any(word in task for word in [
            "clean", "missing values", "preprocess"
        ]):
            return "DataAgent"

        if any(word in task for word in [
            "explore", "eda", "correlation", "pattern"
        ]):
            return "EDAAgent"

        if any(word in task for word in [
            "predict", "classification", "regression", "model"
        ]):
            return "MLAgent"

        if any(word in task for word in [
            "report", "explain"
        ]):
            return "ReportAgent"

        return "DataAgent"
