from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from sybilshield.config import GraphConfig, TrainConfig


@dataclass(slots=True)
class GraphOutput:
    nx_graph: nx.Graph
    pyg_data: Data
    edge_count: int
    density: float


class GraphBuilder:
    def __init__(self, graph_config: GraphConfig, train_config: TrainConfig):
        self.graph_config = graph_config
        self.train_config = train_config

    def build(
        self,
        reviews: pd.DataFrame,
        feature_matrix: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> GraphOutput:
        graph = nx.Graph()
        review_ids = reviews["review_id"].tolist()
        labels = reviews["is_fraud"].astype(int).to_numpy()

        for row in reviews.itertuples(index=False):
            graph.add_node(
                row.review_id,
                user_id=row.user_id,
                product_id=row.product_id,
                rating=int(row.rating),
                is_fraud=int(row.is_fraud),
                ring_id=row.ring_id,
                timestamp=str(row.timestamp),
            )

        self._connect_same_entity(graph, reviews, "user_id", 1.0, "same_user")
        self._connect_same_entity(graph, reviews, "product_id", 0.7, "same_product")
        self._connect_temporal_similarity(graph, reviews, text_embeddings)
        self._connect_semantic_similarity(graph, review_ids, text_embeddings)

        if graph.number_of_edges() == 0:
            for idx in range(len(review_ids) - 1):
                graph.add_edge(review_ids[idx], review_ids[idx + 1], weight=0.1, relation="fallback")

        edge_index, edge_attr = self._to_edge_tensors(graph, review_ids)
        train_mask, val_mask, test_mask = self._make_masks(labels)

        pyg_data = Data(
            x=torch.tensor(feature_matrix, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(labels, dtype=torch.float32),
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            val_mask=torch.tensor(val_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool),
            review_ids=review_ids,
        )
        density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
        return GraphOutput(
            nx_graph=graph,
            pyg_data=pyg_data,
            edge_count=graph.number_of_edges(),
            density=density,
        )

    def _connect_same_entity(
        self,
        graph: nx.Graph,
        reviews: pd.DataFrame,
        column: str,
        weight: float,
        relation: str,
    ) -> None:
        window = (
            self.graph_config.same_user_window
            if column == "user_id"
            else self.graph_config.same_product_window
        )
        for _, group in reviews.sort_values("timestamp").groupby(column):
            review_ids = group["review_id"].tolist()
            if len(review_ids) < 2:
                continue
            for idx, source in enumerate(review_ids):
                upper_bound = min(len(review_ids), idx + 1 + window)
                for target in review_ids[idx + 1 : upper_bound]:
                    self._add_or_update_edge(graph, source, target, weight, relation)

    def _connect_temporal_similarity(
        self,
        graph: nx.Graph,
        reviews: pd.DataFrame,
        text_embeddings: np.ndarray,
    ) -> None:
        indexed = reviews.reset_index(drop=True).copy()
        indexed["timestamp"] = pd.to_datetime(indexed["timestamp"])
        grouped = indexed.groupby("product_id")
        for _, group in grouped:
            if len(group) < 2:
                continue
            ordered = group.sort_values("timestamp")
            positions = ordered.index.to_list()
            for idx, left in enumerate(positions):
                left_row = indexed.iloc[left]
                for right in positions[idx + 1 :]:
                    right_row = indexed.iloc[right]
                    time_gap = abs(
                        (left_row["timestamp"] - right_row["timestamp"]).total_seconds()
                    ) / 3600.0
                    if time_gap > self.graph_config.temporal_window_hours:
                        break
                    similarity = float(np.dot(text_embeddings[left], text_embeddings[right]))
                    if similarity >= self.graph_config.temporal_similarity_threshold:
                        self._add_or_update_edge(
                            graph,
                            left_row["review_id"],
                            right_row["review_id"],
                            1.2,
                            "temporal_semantic_burst",
                        )

    def _connect_semantic_similarity(
        self,
        graph: nx.Graph,
        review_ids: list[str],
        text_embeddings: np.ndarray,
    ) -> None:
        max_neighbors = self.graph_config.max_similarity_neighbors
        neighbor_model = NearestNeighbors(
            n_neighbors=min(len(review_ids), max_neighbors + 1),
            metric="cosine",
        )
        neighbor_model.fit(text_embeddings)
        distances, indices = neighbor_model.kneighbors(text_embeddings)
        for idx, source in enumerate(review_ids):
            for distance, neighbor_idx in zip(distances[idx][1:], indices[idx][1:]):
                score = float(1.0 - distance)
                if score < self.graph_config.text_similarity_threshold:
                    continue
                self._add_or_update_edge(graph, source, review_ids[int(neighbor_idx)], score, "semantic")

    @staticmethod
    def _add_or_update_edge(
        graph: nx.Graph, source: str, target: str, weight: float, relation: str
    ) -> None:
        if source == target:
            return
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += float(weight)
            graph[source][target]["relation"] = f"{graph[source][target]['relation']}|{relation}"
        else:
            graph.add_edge(source, target, weight=float(weight), relation=relation)

    @staticmethod
    def _to_edge_tensors(graph: nx.Graph, review_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        node_index = {node_id: idx for idx, node_id in enumerate(review_ids)}
        edge_pairs: list[list[int]] = []
        edge_weights: list[float] = []
        for source, target, data in graph.edges(data=True):
            source_idx = node_index[source]
            target_idx = node_index[target]
            weight = float(data.get("weight", 1.0))
            edge_pairs.extend([[source_idx, target_idx], [target_idx, source_idx]])
            edge_weights.extend([weight, weight])
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)
        return edge_index, edge_attr

    def _make_masks(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=self.train_config.train_ratio,
            stratify=labels,
            random_state=self.train_config.random_seed,
        )
        adjusted_val_ratio = self.train_config.val_ratio / (1.0 - self.train_config.train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=adjusted_val_ratio,
            stratify=labels[temp_idx],
            random_state=self.train_config.random_seed,
        )
        train_mask = np.zeros(len(labels), dtype=bool)
        val_mask = np.zeros(len(labels), dtype=bool)
        test_mask = np.zeros(len(labels), dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        return train_mask, val_mask, test_mask
