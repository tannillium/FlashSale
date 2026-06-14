from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sybilshield-mpl-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from sybilshield.config import ArtifactConfig


def save_visual_artifacts(
    *,
    artifact_config: ArtifactConfig,
    metrics: dict,
    history: pd.DataFrame,
    reviews: pd.DataFrame,
    graph: nx.Graph,
    model_name: str,
) -> None:
    _save_roc_curve(metrics["test"], artifact_config.roc_curve_path, model_name)
    _save_confusion_matrix(metrics["test"], artifact_config.confusion_matrix_path, model_name)
    _save_fraud_cluster(graph, reviews, artifact_config.fraud_cluster_path)
    _save_training_history(history, artifact_config.training_history_path, model_name)


def _save_roc_curve(test_metrics: dict, output_path: Path, model_name: str) -> None:
    curve = test_metrics.get("roc_curve", {})
    fpr = curve.get("fpr", [])
    tpr = curve.get("tpr", [])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#c2410c", linewidth=2, label=f"{model_name.upper()} ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_confusion_matrix(test_metrics: dict, output_path: Path, model_name: str) -> None:
    matrix = test_metrics.get("confusion_matrix", {})
    grid = np.array(
        [
            [matrix.get("tn", 0), matrix.get("fp", 0)],
            [matrix.get("fn", 0), matrix.get("tp", 0)],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(grid, cmap="Oranges")
    ax.set_title(f"Confusion Matrix ({model_name.upper()})")
    ax.set_xticks([0, 1], labels=["Predicted Legit", "Predicted Fraud"])
    ax.set_yticks([0, 1], labels=["Actual Legit", "Actual Fraud"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(grid[row, col]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_fraud_cluster(graph: nx.Graph, reviews: pd.DataFrame, output_path: Path) -> None:
    suspicious = reviews.sort_values("fraud_probability", ascending=False).head(80)
    subgraph = graph.subgraph(suspicious["review_id"].tolist()).copy()
    positions = nx.spring_layout(subgraph, seed=42)
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(subgraph, positions, alpha=0.25, edge_color="#94a3b8", width=0.8, ax=ax)
    scores = suspicious.set_index("review_id")["fraud_probability"]
    node_colors = [scores.get(node, 0.0) for node in subgraph.nodes()]
    collection = nx.draw_networkx_nodes(
        subgraph,
        positions,
        node_size=90,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        edgecolors="#111827",
        linewidths=0.4,
        ax=ax,
    )
    fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, label="Fraud score")
    ax.set_title("Suspicious Fraud Cluster")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_training_history(history: pd.DataFrame, output_path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(history["epoch"], history["loss"], color="#2563eb", linewidth=2)
    axes[0].set_title(f"{model_name.upper()} Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)

    axes[1].plot(history["epoch"], history["train_f1"], label="Train F1", color="#16a34a", linewidth=2)
    axes[1].plot(history["epoch"], history["val_f1"], label="Validation F1", color="#ea580c", linewidth=2)
    axes[1].set_title("Training History")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
