"""Streamlit web demo for the self-evolving multi-agent analytics framework.

Run from the repository root:
    python -m streamlit run dashboard.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.orchestrator.orchestrator import Orchestrator


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATASET = PROJECT_ROOT / "data" / "titanic.csv"
TEMP_ROOT = Path(tempfile.gettempdir()) / "self_evolving_multi_agent_demo"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="NEXUS AI | Explainable Analytics",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      /* Explicitly set foreground and background so a dark Streamlit theme
         cannot leave light text on the light main panel. */
      .stApp {
        background: #f4f7fb;
        color: #172033;
        --text-color: #172033;
        --background-color: #f4f7fb;
        --secondary-background-color: #ffffff;
        --primary-color: #0f766e;
      }
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="stMainBlockContainer"] {
        background-color: #f4f7fb;
        color: #172033;
      }
      [data-testid="stMain"] [data-testid="stMarkdownContainer"],
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
      [data-testid="stMain"] [data-testid="stExpander"] summary,
      [data-testid="stMain"] [data-testid="stTabs"] button,
      [data-testid="stMain"] label {
        color: #172033;
      }
      [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #64748b;
      }
      [data-testid="stSidebar"] { background: #101827; }
      [data-testid="stSidebar"] * { color: #e8eef7; }
      .hero {
        padding: 1.5rem 1.7rem; border-radius: 20px;
        background: linear-gradient(120deg, #101827 0%, #1e3450 70%, #176e77 100%);
        color: #ffffff; margin-bottom: 1.1rem;
      }
      .hero-kicker { color: #76e5d5; font-size: .78rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase; }
      .hero h1 { color: #ffffff; font-size: 2.15rem; margin: .3rem 0 .45rem 0; }
      .hero p { color: #d8e3f1; margin: 0; max-width: 850px; }
      .section-label { color: #0f766e; font-size: .75rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: .12em; }
      .small-note { color: #64748b; font-size: .85rem; }
      div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e1e8f0;
        padding: 1rem; border-radius: 14px;
      }
      div[data-testid="stMetricLabel"] { color: #64748b; }
      div[data-testid="stMetricValue"] { color: #142136; }
      .finding-card { background: #ffffff; padding: 1rem 1.2rem;
        border: 1px solid #e1e8f0; border-radius: 14px; margin-bottom: .65rem; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_text(value: Any, limit: int = 180) -> str:
    """Convert a value to a concise label suitable for a UI or DOT graph."""
    text = str(value if value is not None else "")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _dot_escape(value: Any) -> str:
    return (
        _safe_text(value, 70)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("<", "\\<")
        .replace(">", "\\>")
    )


def _graph_dot(graph: Any, current_finding_ids: set[str] | None = None) -> str:
    """Build a small Graphviz provenance view from the saved graph snapshot."""
    if graph is None or graph.number_of_nodes() == 0:
        return 'digraph G { label="No graph data yet"; }'

    current_finding_ids = current_finding_ids or set()
    visible = set(current_finding_ids)

    # Include immediate provenance neighbours for current findings.
    for node_id in list(current_finding_ids):
        if node_id not in graph:
            continue
        visible.update(graph.predecessors(node_id))
        visible.update(graph.successors(node_id))
        for parent_id in graph.predecessors(node_id):
            visible.update(graph.predecessors(parent_id))

    # If no current-result IDs are available, show a bounded recent snapshot.
    if not visible:
        visible = set(list(graph.nodes)[-30:])

    visible = {node_id for node_id in visible if node_id in graph}
    node_order = list(graph.nodes)
    if len(visible) > 42:
        wanted = set(current_finding_ids)
        for node_id in reversed(node_order):
            if len(visible) <= 42:
                break
            if node_id not in wanted:
                visible.discard(node_id)

    lines = [
        "digraph KnowledgeGraph {",
        '  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.65"];',
        '  node [style="rounded,filled", fontname="Arial", fontsize="10", color="#d9e2ef", fontcolor="#172033", margin="0.12,0.08"];',
        '  edge [fontname="Arial", fontsize="8", color="#8292a8", fontcolor="#526176", arrowsize="0.7"];',
    ]

    shapes = {
        "dataset": ("folder", "#dbeafe"),
        "feature": ("ellipse", "#e0e7ff"),
        "finding": ("box", "#ccfbf1"),
        "evidence": ("note", "#fef3c7"),
        "model": ("component", "#f3e8ff"),
        "execution": ("diamond", "#fee2e2"),
    }

    for node_id in node_order:
        if node_id not in visible:
            continue
        attrs = graph.nodes[node_id]
        node_type = _safe_text(attrs.get("type", "node"), 30).lower()
        shape, fill = shapes.get(node_type, ("ellipse", "#f1f5f9"))
        title = (
            attrs.get("description")
            or attrs.get("name")
            or attrs.get("dataset")
            or attrs.get("task")
            or node_id
        )
        label = f"{node_type.title()}\\n{_dot_escape(title)}"
        lines.append(
            f'  "{_dot_escape(node_id)}" [label="{label}", shape="{shape}", fillcolor="{fill}"];'
        )

    for source, target, edge_attrs in graph.edges(data=True):
        if source not in visible or target not in visible:
            continue
        relation = _dot_escape(edge_attrs.get("relation", "related_to"))
        lines.append(
            f'  "{_dot_escape(source)}" -> "{_dot_escape(target)}" [label="{relation}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def _get_evidence_description(graph: Any, evidence_id: str) -> str:
    if graph is None or evidence_id not in graph:
        return "Evidence details are unavailable in the current graph snapshot."
    attrs = graph.nodes[evidence_id]
    return _safe_text(attrs.get("description") or attrs.get("name") or evidence_id, 500)


def _task_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in output.get("results", []):
        checks = item.get("verification", [])
        rows.append(
            {
                "Task": _safe_text(item.get("task") or "Analysis task", 80),
                "Task ID": item.get("task_id") or "—",
                "Findings": len(item.get("finding_ids", [])),
                "Structurally valid": sum(
                    check.get("structural_verified") is True for check in checks
                ),
                "Checks": len(checks),
                "Execution successful": item.get("evaluation", {}).get("successful", False),
            }
        )
    return rows


# Keep a stable temporary graph file across Streamlit reruns in this browser session.
if "demo_session_id" not in st.session_state:
    st.session_state.demo_session_id = uuid.uuid4().hex

session_id = st.session_state.demo_session_id


# Sidebar: dataset selection and project status.
with st.sidebar:
    st.markdown("## 🔷 NEXUS AI")
    st.caption("Explainable multi-agent analytics")
    st.markdown("---")
    st.markdown("### Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Leave this empty to use the included Titanic example dataset.",
    )

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        dataset_key = hashlib.sha256(upload_bytes).hexdigest()[:16]
        dataset_path = TEMP_ROOT / f"{session_id}_{dataset_key}_{Path(uploaded_file.name).name}"
        if not dataset_path.exists() or hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset_key:
            # Compare with the complete file digest separately; the shortened key is for naming only.
            full_digest = hashlib.sha256(upload_bytes).hexdigest()
            dataset_path = TEMP_ROOT / f"{session_id}_{full_digest[:16]}_{Path(uploaded_file.name).name}"
            if not dataset_path.exists() or hashlib.sha256(dataset_path.read_bytes()).hexdigest() != full_digest:
                dataset_path.write_bytes(upload_bytes)
        dataset_label = uploaded_file.name
        dataset_fingerprint = dataset_key
    else:
        dataset_path = SAMPLE_DATASET
        dataset_label = "Titanic sample dataset"
        dataset_fingerprint = "sample_titanic"

    st.markdown(f"**Selected:** {_safe_text(dataset_label, 45)}")
    if not dataset_path.exists():
        st.error(f"Dataset not found: {dataset_path}")

    st.markdown("---")
    st.markdown("### Current capabilities")
    st.markdown("- EDA: correlations, missingness, outliers")
    st.markdown("- Knowledge graph and evidence links")
    st.markdown("- Execution history and workflow recommendations")
    st.markdown("---")
    st.caption("ML-agent and report-agent execution are not yet integrated.")


# Stable graph path per browser session and dataset, isolated in the OS temp directory.
graph_dir = TEMP_ROOT / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / f"{session_id}_{dataset_fingerprint}.json"


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Self-Evolving Multi-Agent Framework</div>
      <h1>Explainable Data Analytics</h1>
      <p>Explore a dataset, inspect the findings and evidence, and trace how the analysis is represented in a persistent knowledge graph.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("analysis_form"):
    request_text = st.text_area(
        "What would you like to investigate?",
        value="Explore correlations and patterns in this dataset",
        height=85,
        help="The current integrated execution path supports exploratory data analysis.",
    )
    run_clicked = st.form_submit_button(
        "▶  Run analysis",
        type="primary",
        use_container_width=True,
    )

if run_clicked:
    if not dataset_path.exists():
        st.error("The selected dataset could not be found. Choose a CSV file and try again.")
    elif not request_text.strip():
        st.warning("Enter a request before running the analysis.")
    else:
        try:
            with st.spinner("Running the EDA pipeline and updating the knowledge graph…"):
                orchestrator = Orchestrator(
                    dataset_path=str(dataset_path),
                    graph_path=graph_path,
                )
                output = orchestrator.process_request(request_text.strip())
                st.session_state.last_output = output
                st.session_state.last_graph = orchestrator.knowledge_graph.get_graph()
                st.session_state.last_dataset_label = dataset_label
                st.session_state.last_request = request_text.strip()
                st.session_state.last_graph_path = str(graph_path)
                st.session_state.last_error = None
        except Exception as exc:  # Display a concise UI error; details are available to expand.
            st.session_state.last_error = f"{type(exc).__name__}: {exc}"

if st.session_state.get("last_error"):
    st.error("The analysis did not complete. Check the technical details below.")
    with st.expander("Technical details"):
        st.code(st.session_state.last_error)

output = st.session_state.get("last_output")
graph_snapshot = st.session_state.get("last_graph")

if output is None:
    st.info("Choose a CSV dataset, enter a request, and click **Run analysis** to begin.")
    st.markdown("#### What this demo shows")
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown("**01 · Analyze**")
        st.write("Execute the currently integrated exploratory data analysis pipeline.")
    with feature_cols[1]:
        st.markdown("**02 · Explain**")
        st.write("Inspect each finding alongside the evidence linked to it.")
    with feature_cols[2]:
        st.markdown("**03 · Learn from history**")
        st.write("View recorded executions and the workflow recommendations they inform.")
    st.stop()


# Result header and headline metrics.
st.markdown('<div class="section-label">Latest run</div>', unsafe_allow_html=True)
status = output.get("status", "unknown")
if status == "completed":
    st.success("Workflow completed. Structural validation passed for the findings reported as valid.")
elif status == "completed_with_warnings":
    st.warning("Workflow finished with warnings. Some results did not pass structural checks.")
elif status == "completed_with_errors":
    st.error("Workflow finished with errors. Review the task-level details below.")
elif status == "not_implemented":
    st.warning(output.get("message", "This agent's execution workflow is not yet connected."))
else:
    st.info(f"Workflow status: {status}")

results = output.get("results", [])
finding_count = sum(len(item.get("finding_ids", [])) for item in results)
verification_checks = [check for item in results for check in item.get("verification", [])]
structural_count = sum(check.get("structural_verified") is True for check in verification_checks)
execution_success_count = sum(item.get("evaluation", {}).get("successful") is True for item in results)
execution_count = 0
if graph_snapshot is not None:
    execution_count = sum(
        1 for _, attrs in graph_snapshot.nodes(data=True)
        if attrs.get("type") == "execution"
    )

metric_cols = st.columns(4)
metric_cols[0].metric("Analysis tasks", output.get("task_count", len(results)))
metric_cols[1].metric("Findings", finding_count)
metric_cols[2].metric("Structural checks passed", f"{structural_count}/{len(verification_checks)}")
metric_cols[3].metric("Recorded executions", execution_count)

st.caption(
    f"Dataset: {st.session_state.get('last_dataset_label', dataset_label)}  ·  "
    f"Request: {st.session_state.get('last_request', '')}"
)


tab_overview, tab_findings, tab_graph, tab_history = st.tabs(
    ["📊 Overview", "🔎 Findings & evidence", "🕸️ Knowledge graph", "🔁 Execution history"]
)

with tab_overview:
    st.subheader("Analysis overview")
    if results:
        rows = _task_rows(output)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No completed task results were returned for this request.")

    st.subheader("Ranked insights")
    insights = output.get("insights", [])
    if not insights:
        st.caption("The pipeline did not return a separate ranked-insights list for this run.")
    else:
        for index, insight in enumerate(insights[:10], start=1):
            if isinstance(insight, dict):
                title = (
                    insight.get("title")
                    or insight.get("description")
                    or insight.get("finding")
                    or insight.get("insight")
                    or f"Insight {index}"
                )
                with st.expander(f"{index}. {_safe_text(title, 130)}"):
                    st.json(insight)
            else:
                st.markdown(f"**{index}.** {insight}")

    errors = output.get("errors", [])
    if errors:
        with st.expander(f"Task errors ({len(errors)})"):
            st.json(errors)

with tab_findings:
    st.subheader("Findings and supporting evidence")
    if not results:
        st.info("Run an analysis to see findings here.")
    for item in results:
        finding_ids = item.get("finding_ids", [])
        findings = item.get("findings", [])
        checks = item.get("verification", [])
        task_title = _safe_text(item.get("task") or item.get("task_id") or "Analysis task", 90)
        st.markdown(f"#### {task_title}")
        for index, finding_id in enumerate(finding_ids):
            finding_text = findings[index] if index < len(findings) else finding_id
            check = checks[index] if index < len(checks) else {}
            with st.expander(f"{_safe_text(finding_id, 60)} · {_safe_text(finding_text, 120)}"):
                st.write(finding_text)
                if check.get("structural_verified") is True:
                    st.success("Structural validation passed")
                else:
                    st.warning("One or more structural checks did not pass")
                st.info(
                    "Scientific support has not been assessed by the current FactChecker. "
                    "A connected evidence item is not, by itself, proof that a claim is true."
                )
                if check.get("evidence"):
                    st.markdown("**Linked evidence**")
                    for evidence_id in check["evidence"]:
                        st.markdown(
                            f"- **{_safe_text(evidence_id, 70)}:** "
                            f"{_get_evidence_description(graph_snapshot, evidence_id)}"
                        )
                if check.get("models"):
                    st.markdown("**Linked models**")
                    for model_id in check["models"]:
                        st.write(model_id)
                with st.expander("Structural checks (technical details)"):
                    st.json(check.get("checks", {}))
        evaluation = item.get("evaluation")
        if evaluation:
            with st.expander(f"Execution evaluation · {task_title}"):
                st.json(evaluation)

with tab_graph:
    st.subheader("Knowledge graph")
    if graph_snapshot is None or graph_snapshot.number_of_nodes() == 0:
        st.info("The graph will appear after a successful analysis run.")
    else:
        current_finding_ids = {
            finding_id
            for item in results
            for finding_id in item.get("finding_ids", [])
        }
        graph_cols = st.columns(3)
        graph_cols[0].metric("Nodes", graph_snapshot.number_of_nodes())
        graph_cols[1].metric("Relationships", graph_snapshot.number_of_edges())
        graph_cols[2].metric(
            "Node types",
            len({attrs.get("type", "unknown") for _, attrs in graph_snapshot.nodes(data=True)}),
        )
        try:
            st.graphviz_chart(_graph_dot(graph_snapshot, current_finding_ids), width="stretch")
        except Exception as exc:
            st.warning("The graph visualization could not be rendered. A node and edge summary is available below.")
            st.caption(_safe_text(exc, 300))

        with st.expander("Graph nodes"):
            node_rows = []
            for node_id, attrs in graph_snapshot.nodes(data=True):
                node_rows.append(
                    {
                        "ID": node_id,
                        "Type": attrs.get("type", "unknown"),
                        "Label": _safe_text(
                            attrs.get("description")
                            or attrs.get("name")
                            or attrs.get("dataset")
                            or attrs.get("task")
                            or node_id,
                            140,
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(node_rows), width="stretch", hide_index=True)

        with st.expander("Graph relationships"):
            edge_rows = [
                {
                    "Source": source,
                    "Relationship": attrs.get("relation", "related_to"),
                    "Target": target,
                }
                for source, target, attrs in graph_snapshot.edges(data=True)
            ]
            st.dataframe(pd.DataFrame(edge_rows), width="stretch", hide_index=True)

with tab_history:
    st.subheader("Execution history and self-evolution")
    st.write(
        "The engine records execution outcomes and uses relevant task/dataset history "
        "to inform future workflow recommendations. This is history-based adaptation, "
        "not autonomous model retraining."
    )
    if graph_snapshot is None:
        st.info("Execution history becomes available after a run.")
    else:
        history = []
        for node_id, attrs in graph_snapshot.nodes(data=True):
            if attrs.get("type") == "execution":
                history.append(
                    {
                        "Execution ID": node_id,
                        "Agent": attrs.get("agent", "—"),
                        "Task": attrs.get("task", "—"),
                        "Dataset": attrs.get("dataset", "—"),
                        "Successful": attrs.get("successful", "—"),
                        "Confidence": attrs.get("confidence", "—"),
                        "Model performance": attrs.get("model_performance", "—"),
                    }
                )
        if history:
            st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
        else:
            st.info("No execution-history nodes were found in this graph yet.")

    st.markdown("#### Scope and limitations")
    st.markdown(
        "- The EDA path is currently integrated for correlations, missingness, and outliers.\n"
        "- The FactChecker validates graph structure and metadata, not scientific entailment.\n"
        "- Confidence is a provisional sample-size heuristic, not a calibrated probability.\n"
        "- ML-agent and report-agent execution pipelines are not yet connected to the orchestrator."
    )

st.markdown("---")
st.caption("NEXUS AI · Research prototype · Evidence-linked analytics and knowledge-graph reasoning")
