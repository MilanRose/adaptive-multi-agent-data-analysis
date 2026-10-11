
from app.schemas import AgentResult

# Test explicit feature names
result = AgentResult(
    agent="EDAAgent",
    dataset="customer_churn.csv",
    task="correlation_analysis",
    findings=["MonthlyCharges is associated with churn"],
    features=["MonthlyCharges", "tenure"],
    confidence=0.84
)

assert result.features == ["MonthlyCharges", "tenure"]

# Test backward compatibility: features are optional
old_result = AgentResult(
    agent="EDAAgent",
    dataset="customer_churn.csv",
    task="correlation_analysis",
    findings=["A finding without explicit features"],
    confidence=0.80
)

assert old_result.features == []

print("FEATURE SCHEMA TEST PASSED")