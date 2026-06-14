from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class DataConfig:
    num_reviews: int = 10000
    num_users: int = 1800
    num_products: int = 320
    fraud_ratio: float = 0.24
    num_rings: int = 12
    max_reviews_per_user: int = 24
    start_date: str = "2025-01-01"
    days_span: int = 240
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    random_seed: int = 42


@dataclass(slots=True)
class GraphConfig:
    text_similarity_threshold: float = 0.82
    temporal_window_hours: float = 36.0
    temporal_similarity_threshold: float = 0.78
    max_similarity_neighbors: int = 8
    same_user_window: int = 4
    same_product_window: int = 6


@dataclass(slots=True)
class TrainConfig:
    model_name: str = "gat"
    hidden_dim: int = 64
    dropout: float = 0.35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 12
    heads: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    threshold: float = 0.5
    device: str = "cpu"
    random_seed: int = 42


@dataclass(slots=True)
class ArtifactConfig:
    root_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")

    @property
    def reviews_path(self) -> Path:
        return self.root_dir / "reviews_with_predictions.csv"

    @property
    def synthetic_reviews_path(self) -> Path:
        return self.root_dir / "synthetic_reviews.csv"

    @property
    def metrics_path(self) -> Path:
        return self.root_dir / "metrics.json"

    @property
    def history_path(self) -> Path:
        return self.root_dir / "history.csv"

    @property
    def graph_path(self) -> Path:
        return self.root_dir / "graph.pkl"

    @property
    def checkpoint_path(self) -> Path:
        return self.root_dir / "checkpoint.pt"

    @property
    def run_config_path(self) -> Path:
        return self.root_dir / "run_config.json"

    @property
    def roc_curve_path(self) -> Path:
        return self.root_dir / "roc_curve.png"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.root_dir / "confusion_matrix.png"

    @property
    def fraud_cluster_path(self) -> Path:
        return self.root_dir / "fraud_cluster.png"

    @property
    def training_history_path(self) -> Path:
        return self.root_dir / "training_history.png"

    @property
    def model_comparison_path(self) -> Path:
        return self.root_dir / "model_comparison.json"

    def model_dir(self, model_name: str) -> Path:
        return self.root_dir / "models" / model_name.lower()
