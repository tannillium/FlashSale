from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from sybilshield.config import ArtifactConfig, DataConfig, GraphConfig, TrainConfig
from sybilshield.data import SyntheticDatasetGenerator
from sybilshield.evaluation import compute_classification_metrics
from sybilshield.features import FeatureEngineer
from sybilshield.graph import GraphBuilder
from sybilshield.models import GATReviewDetector, GCNReviewDetector
from sybilshield.utils import ensure_dir, save_json, save_pickle, set_seed
from sybilshield.visualization import save_visual_artifacts


@dataclass(slots=True)
class TrainingResult:
    reviews: pd.DataFrame
    metrics: dict
    history: pd.DataFrame
    graph_summary: dict


class TrainingPipeline:
    def __init__(
        self,
        data_config: DataConfig | None = None,
        graph_config: GraphConfig | None = None,
        train_config: TrainConfig | None = None,
        artifact_config: ArtifactConfig | None = None,
    ):
        self.data_config = data_config or DataConfig()
        self.graph_config = graph_config or GraphConfig()
        self.train_config = train_config or TrainConfig()
        self.artifact_config = artifact_config or ArtifactConfig()

    def run(self) -> TrainingResult:
        ensure_dir(self.artifact_config.root_dir)
        set_seed(self.train_config.random_seed)

        generator = SyntheticDatasetGenerator(self.data_config)
        reviews = generator.generate()
        reviews.to_csv(self.artifact_config.synthetic_reviews_path, index=False)

        feature_engineer = FeatureEngineer(self.data_config)
        feature_output = feature_engineer.transform(reviews)

        graph_builder = GraphBuilder(self.graph_config, self.train_config)
        graph_output = graph_builder.build(
            reviews=feature_output.dataframe,
            feature_matrix=feature_output.feature_matrix,
            text_embeddings=feature_output.text_embeddings,
        )

        data = graph_output.pyg_data
        device = torch.device(self.train_config.device)
        data = data.to(device)
        model = self._build_model(input_dim=data.x.shape[1]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = -1.0
        best_state = None
        wait = 0
        history_rows: list[dict] = []

        for epoch in range(1, self.train_config.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_metrics = self._evaluate_split(model, data, "train")
            val_metrics = self._evaluate_split(model, data, "val")
            current_val_auc = val_metrics.get("roc_auc") or 0.0
            history_rows.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "train_f1": train_metrics["f1"],
                    "val_f1": val_metrics["f1"],
                    "val_roc_auc": current_val_auc,
                }
            )

            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.train_config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()

        history = pd.DataFrame(history_rows)
        full_reviews = feature_output.dataframe.copy()
        full_reviews["fraud_probability"] = probabilities
        full_reviews["predicted_label"] = (
            full_reviews["fraud_probability"] >= self.train_config.threshold
        ).astype(int)

        train_metrics = self._evaluate_split(model, data, "train")
        val_metrics = self._evaluate_split(model, data, "val")
        test_metrics = self._evaluate_split(model, data, "test")
        metrics = {
            "model_name": self.train_config.model_name.lower(),
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "graph": {
                "nodes": int(graph_output.nx_graph.number_of_nodes()),
                "edges": int(graph_output.edge_count),
                "density": float(graph_output.density),
            },
            "dataset": {
                "num_reviews": int(len(full_reviews)),
                "fraud_reviews": int(full_reviews["is_fraud"].sum()),
                "fraud_ratio": float(full_reviews["is_fraud"].mean()),
                "num_users": int(full_reviews["user_id"].nunique()),
                "num_products": int(full_reviews["product_id"].nunique()),
                "num_campaigns": int(full_reviews["campaign_id"].nunique()),
            },
        }

        self._save_artifacts(full_reviews, history, metrics, graph_output.nx_graph, model)
        return TrainingResult(
            reviews=full_reviews,
            metrics=metrics,
            history=history,
            graph_summary=metrics["graph"],
        )

    def _build_model(self, input_dim: int) -> torch.nn.Module:
        if self.train_config.model_name.lower() == "gcn":
            return GCNReviewDetector(
                input_dim=input_dim,
                hidden_dim=self.train_config.hidden_dim,
                dropout=self.train_config.dropout,
            )
        return GATReviewDetector(
            input_dim=input_dim,
            hidden_dim=self.train_config.hidden_dim,
            dropout=self.train_config.dropout,
            heads=self.train_config.heads,
        )

    def _evaluate_split(self, model: torch.nn.Module, data: torch.Tensor, split: str) -> dict:
        mask = getattr(data, f"{split}_mask")
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
            labels = data.y[mask].detach().cpu().numpy()
        return compute_classification_metrics(labels, scores, threshold=self.train_config.threshold)

    def _save_artifacts(
        self,
        reviews: pd.DataFrame,
        history: pd.DataFrame,
        metrics: dict,
        graph,
        model: torch.nn.Module,
    ) -> None:
        reviews.to_csv(self.artifact_config.reviews_path, index=False)
        history.to_csv(self.artifact_config.history_path, index=False)
        save_json(self.artifact_config.metrics_path, metrics)
        save_pickle(self.artifact_config.graph_path, graph)
        torch.save(model.state_dict(), self.artifact_config.checkpoint_path)
        save_visual_artifacts(
            artifact_config=self.artifact_config,
            metrics=metrics,
            history=history,
            reviews=reviews,
            graph=graph,
            model_name=self.train_config.model_name,
        )
        save_json(
            self.artifact_config.run_config_path,
            {
                "data_config": self.data_config,
                "graph_config": self.graph_config,
                "train_config": self.train_config,
            },
        )
