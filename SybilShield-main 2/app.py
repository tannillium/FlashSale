from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sybilshield.config import ArtifactConfig, DataConfig, TrainConfig
from sybilshield.training import TrainingPipeline
from sybilshield.utils import save_json


ARTIFACTS = ArtifactConfig()


def load_reviews(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_graph(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.18), transparent 30%),
                linear-gradient(180deg, #fffdf7 0%, #fff7ed 48%, #ffffff 100%);
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(194, 65, 12, 0.12);
            box-shadow: 0 20px 60px rgba(120, 53, 15, 0.08);
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_graph_figure(graph: nx.Graph, reviews: pd.DataFrame) -> go.Figure:
    suspicious = reviews.sort_values("fraud_probability", ascending=False).head(70)
    subgraph = graph.subgraph(suspicious["review_id"].tolist()).copy()
    positions = nx.spring_layout(subgraph, seed=42)

    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in subgraph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    lookup = suspicious.set_index("review_id")
    node_x: list[float] = []
    node_y: list[float] = []
    colors: list[float] = []
    texts: list[str] = []
    for node in subgraph.nodes():
        x, y = positions[node]
        row = lookup.loc[node]
        node_x.append(x)
        node_y.append(y)
        colors.append(float(row["fraud_probability"]))
        texts.append(
            f"{node}<br>User: {row['user_id']}<br>Product: {row['product_id']}<br>"
            f"Fraud score: {row['fraud_probability']:.3f}<br>Attack: {row['attack_type']}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 0.8, "color": "#94a3b8"},
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={
                "size": 11,
                "color": colors,
                "colorscale": "YlOrRd",
                "showscale": True,
                "colorbar": {"title": "Fraud score"},
                "line": {"width": 0.7, "color": "#111827"},
            },
            text=texts,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=540,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor="rgba(255,255,255,0.85)",
        plot_bgcolor="rgba(255,255,255,0.85)",
        title="Suspicious Cluster Explorer",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def render_roc_figure(metrics: dict) -> go.Figure:
    curve = metrics["test"]["roc_curve"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["fpr"],
            y=curve["tpr"],
            mode="lines",
            name="ROC",
            line={"color": "#c2410c", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "#94a3b8", "dash": "dash"},
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=360,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
    )
    return fig


def render_confusion_matrix(metrics: dict) -> go.Figure:
    matrix = metrics["test"]["confusion_matrix"]
    grid = [
        [matrix["tn"], matrix["fp"]],
        [matrix["fn"], matrix["tp"]],
    ]
    fig = px.imshow(
        grid,
        text_auto=True,
        color_continuous_scale="Oranges",
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        aspect="auto",
    )
    fig.update_layout(title="Confusion Matrix", height=360, margin={"l": 20, "r": 20, "t": 50, "b": 10})
    return fig


def render_score_distribution(reviews: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        reviews,
        x="fraud_probability",
        nbins=25,
        color="is_fraud",
        color_discrete_map={0: "#2563eb", 1: "#c2410c"},
        barmode="overlay",
        opacity=0.8,
        labels={"fraud_probability": "Fraud score", "is_fraud": "Label"},
        title="Fraud Score Distribution",
    )
    fig.update_layout(height=360, margin={"l": 20, "r": 20, "t": 50, "b": 10})
    return fig


def render_model_comparison(comparison: dict | None) -> go.Figure | None:
    if not comparison or not comparison.get("models"):
        return None
    df = pd.DataFrame(comparison["models"])
    melted = df.melt(id_vars="model_name", value_vars=["roc_auc", "precision", "recall", "f1"])
    fig = px.bar(
        melted,
        x="variable",
        y="value",
        color="model_name",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=["#2563eb", "#c2410c"],
        title="Model Comparison",
    )
    fig.update_layout(height=380, margin={"l": 20, "r": 20, "t": 50, "b": 10}, yaxis_range=[0, 1.05])
    return fig


def main() -> None:
    st.set_page_config(page_title="SybilShield", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">SybilShield</h1>
            <p style="margin:0.4rem 0 0 0;">
                Portfolio-ready fake review ring detection with graph neural networks, coordinated fraud simulation,
                and artifact-backed model monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run Controls")
        model_name = st.selectbox("Primary model", ["gat", "gcn"], index=0)
        num_reviews = st.slider("Synthetic reviews", min_value=2000, max_value=20000, value=10000, step=1000)
        epochs = st.slider("Epochs", min_value=10, max_value=80, value=40, step=5)
        compare_models = st.checkbox("Train both models for comparison", value=True)
        retrain = st.button("Generate Fresh Artifacts", use_container_width=True)

    if retrain:
        with st.spinner("Training SybilShield pipeline..."):
            if compare_models:
                comparison_rows: list[dict] = []
                candidate_models = [candidate for candidate in ["gcn", "gat"] if candidate != model_name] + [model_name]
                for candidate in candidate_models:
                    result = TrainingPipeline(
                        data_config=DataConfig(num_reviews=num_reviews),
                        train_config=TrainConfig(model_name=candidate, epochs=epochs),
                        artifact_config=ARTIFACTS,
                    ).run()
                    comparison_rows.append(
                        {
                            "model_name": candidate,
                            "roc_auc": result.metrics["test"]["roc_auc"],
                            "precision": result.metrics["test"]["precision"],
                            "recall": result.metrics["test"]["recall"],
                            "f1": result.metrics["test"]["f1"],
                            "accuracy": result.metrics["test"]["accuracy"],
                        }
                    )
                save_json(
                    ARTIFACTS.model_comparison_path,
                    {"generated_from": candidate_models, "models": comparison_rows},
                )
            else:
                TrainingPipeline(
                    data_config=DataConfig(num_reviews=num_reviews),
                    train_config=TrainConfig(model_name=model_name, epochs=epochs),
                    artifact_config=ARTIFACTS,
                ).run()
        st.success("Artifacts refreshed.")

    reviews = load_reviews(ARTIFACTS.reviews_path)
    metrics = load_json(ARTIFACTS.metrics_path)
    graph = load_graph(ARTIFACTS.graph_path)
    comparison = load_json(ARTIFACTS.model_comparison_path)

    if reviews is None or metrics is None or graph is None:
        st.info("No artifacts found yet. Use the sidebar to generate the full artifact suite.")
        return

    test_metrics = metrics["test"]
    graph_metrics = metrics["graph"]
    dataset_metrics = metrics["dataset"]
    cards = st.columns(6)
    cards[0].metric("Model", metrics["model_name"].upper())
    cards[1].metric("ROC-AUC", f"{(test_metrics['roc_auc'] or 0.0):.3f}")
    cards[2].metric("Precision", f"{test_metrics['precision']:.3f}")
    cards[3].metric("Recall", f"{test_metrics['recall']:.3f}")
    cards[4].metric("F1", f"{test_metrics['f1']:.3f}")
    cards[5].metric("Fraud Reviews", f"{dataset_metrics['fraud_reviews']:,}")

    overview, diagnostics, samples, saved_assets = st.tabs(
        ["Overview", "Diagnostics", "Suspicious Samples", "Saved Assets"]
    )

    with overview:
        left, right = st.columns([1.55, 1])
        with left:
            st.plotly_chart(render_graph_figure(graph, reviews), use_container_width=True)
        with right:
            st.subheader("Dataset Snapshot")
            st.json(
                {
                    "reviews": dataset_metrics["num_reviews"],
                    "users": dataset_metrics["num_users"],
                    "products": dataset_metrics["num_products"],
                    "campaigns": dataset_metrics["num_campaigns"],
                    "graph_edges": graph_metrics["edges"],
                    "graph_density": round(graph_metrics["density"], 5),
                }
            )
            st.subheader("Attack Mix")
            attack_summary = (
                reviews.groupby("attack_type", as_index=False)
                .agg(reviews=("review_id", "count"), avg_score=("fraud_probability", "mean"))
                .sort_values("avg_score", ascending=False)
            )
            st.dataframe(attack_summary, use_container_width=True)

    with diagnostics:
        top_left, top_right = st.columns(2)
        bottom_left, bottom_right = st.columns(2)
        with top_left:
            st.plotly_chart(render_roc_figure(metrics), use_container_width=True)
        with top_right:
            st.plotly_chart(render_confusion_matrix(metrics), use_container_width=True)
        with bottom_left:
            st.plotly_chart(render_score_distribution(reviews), use_container_width=True)
        with bottom_right:
            comparison_fig = render_model_comparison(comparison)
            if comparison_fig is not None:
                st.plotly_chart(comparison_fig, use_container_width=True)
            else:
                st.info("Run both GCN and GAT to unlock model comparison.")

    with samples:
        st.subheader("Top Suspicious Reviews")
        st.dataframe(
            reviews.sort_values("fraud_probability", ascending=False)[
                [
                    "review_id",
                    "user_id",
                    "product_id",
                    "rating",
                    "fraud_probability",
                    "ring_id",
                    "attack_type",
                    "review_text",
                ]
            ].head(25),
            use_container_width=True,
        )
        st.subheader("Campaign Breakdown")
        campaign_summary = (
            reviews.groupby(["campaign_id", "attack_type"], as_index=False)
            .agg(
                reviews=("review_id", "count"),
                avg_score=("fraud_probability", "mean"),
                fraud_rate=("is_fraud", "mean"),
            )
            .sort_values("avg_score", ascending=False)
        )
        st.dataframe(campaign_summary, use_container_width=True)

    with saved_assets:
        st.image(str(ARTIFACTS.roc_curve_path), caption="Saved ROC Curve", use_container_width=True)
        st.image(str(ARTIFACTS.confusion_matrix_path), caption="Saved Confusion Matrix", use_container_width=True)
        st.image(str(ARTIFACTS.fraud_cluster_path), caption="Saved Fraud Cluster", use_container_width=True)
        st.image(str(ARTIFACTS.training_history_path), caption="Saved Training History", use_container_width=True)


if __name__ == "__main__":
    main()
