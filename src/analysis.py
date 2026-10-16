from scipy import stats
import pandas as pd


def execute_correlation_task(df: pd.DataFrame, task: dict) -> dict:
    """
    Recomputes the Pearson correlation for the task's two columns,
    including the p-value (a measure of statistical significance).
    """
    col1, col2 = task["columns"]
    clean_df = df[[col1, col2]].dropna()

    r_value, p_value = stats.pearsonr(clean_df[col1], clean_df[col2])

    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "columns": [col1, col2],
        "method": "Pearson correlation",
        "value": round(r_value, 3),
        "p_value": round(p_value, 5),
        "sample_size": len(clean_df),
    }


def execute_missingness_task(df: pd.DataFrame, task: dict) -> dict:
    """
    Recomputes the missing-value count and percentage for the task's column.
    """
    column = task["columns"][0]
    n_missing = df[column].isnull().sum()
    pct_missing = round((n_missing / len(df)) * 100, 2)

    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "columns": [column],
        "method": "Missing value count",
        "n_missing": int(n_missing),
        "pct_missing": pct_missing,
        "sample_size": len(df),
    }


def execute_outlier_task(df: pd.DataFrame, task: dict) -> dict:
    """
    Recomputes outlier count/percentage for the task's column using IQR.
    """
    column = task["columns"][0]
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    pct_outliers = round((len(outliers) / len(df)) * 100, 2)

    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "columns": [column],
        "method": "IQR outlier detection",
        "n_outliers": len(outliers),
        "pct_outliers": pct_outliers,
        "bounds": [round(lower_bound, 2), round(upper_bound, 2)],
        "sample_size": len(df),
    }


def execute_task(df: pd.DataFrame, task: dict) -> dict:
    """
    Routes a task to the correct execution function based on its type.
    """
    task_type = task["task"]

    if task_type == "investigate_correlation":
        return execute_correlation_task(df, task)
    elif task_type == "investigate_missingness":
        return execute_missingness_task(df, task)
    elif task_type == "investigate_outliers":
        return execute_outlier_task(df, task)
    else:
        return {"task_id": task["task_id"], "error": f"Unknown task type: {task_type}"}


if __name__ == "__main__":
    from data_loader import load_dataset
    from eda import compute_correlations, detect_outliers_iqr
    from profiler import profile_dataset
    from insights import (
        detect_correlation_insights,
        detect_missingness_insights,
        detect_outlier_insights,
        rank_insights,
    )
    from planner import build_task_plan

    df = load_dataset("data/titanic.csv")
    profile = profile_dataset(df)
    correlations = compute_correlations(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_results = [detect_outliers_iqr(df, col) for col in numeric_cols]

    corr_insights = detect_correlation_insights(correlations, threshold=0.3)
    missing_insights = detect_missingness_insights(profile["missing_pct"], threshold=20.0)
    outlier_insights = detect_outlier_insights(outlier_results, threshold=5.0)

    all_insights = corr_insights + missing_insights + outlier_insights
    ranked_insights = rank_insights(all_insights)
    task_plan = build_task_plan(ranked_insights)

    print(f"=== Executing {len(task_plan)} tasks ===\n")
    execution_results = []
    for task in task_plan:
        result = execute_task(df, task)
        execution_results.append(result)
        print(result)