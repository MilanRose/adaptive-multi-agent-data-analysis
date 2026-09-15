from app.schemas import (
    AgentResult,
    Evidence,
    ModelResult
)


result = AgentResult(

    agent="EDAAgent",

    dataset="customer_churn.csv",

    task="correlation_analysis",

    findings=[
        "MonthlyCharges is associated with churn"
    ],

    evidence=[
        Evidence(
            description="Pearson correlation = 0.42",
            method="Pearson correlation",
            value=0.42
        )
    ],

    model=ModelResult(
        name="Random Forest",
        metrics={
            "accuracy": 0.87
        }
    ),

    confidence=0.84
)


print("AGENT RESULT:")
print(result.model_dump())