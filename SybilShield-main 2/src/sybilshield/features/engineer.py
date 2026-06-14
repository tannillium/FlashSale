from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler

from sybilshield.config import DataConfig


@dataclass(slots=True)
class FeatureOutput:
    feature_matrix: np.ndarray
    text_embeddings: np.ndarray
    feature_names: list[str]
    dataframe: pd.DataFrame


class FeatureEngineer:
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.encoder = None
        self.vectorizer = HashingVectorizer(
            n_features=256,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def transform(self, reviews: pd.DataFrame) -> FeatureOutput:
        df = reviews.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["review_length"] = df["review_text"].str.len()
        df["word_count"] = df["review_text"].str.split().str.len()
        df["exclamation_count"] = df["review_text"].str.count("!")
        df["uppercase_ratio"] = df["review_text"].apply(self._uppercase_ratio)
        df["rating_deviation"] = (df["rating"] - df["rating"].mean()).abs()

        user_counts = df.groupby("user_id")["review_id"].transform("count")
        product_counts = df.groupby("product_id")["review_id"].transform("count")
        df["user_review_count"] = user_counts
        df["product_review_count"] = product_counts

        df["user_mean_rating"] = df.groupby("user_id")["rating"].transform("mean")
        df["product_mean_rating"] = df.groupby("product_id")["rating"].transform("mean")
        df["hours_since_start"] = (
            (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0
        )
        df["user_burstiness_hours"] = self._burstiness(df, "user_id")
        df["product_burstiness_hours"] = self._burstiness(df, "product_id")

        numeric_columns = [
            "rating",
            "review_length",
            "word_count",
            "exclamation_count",
            "uppercase_ratio",
            "rating_deviation",
            "user_review_count",
            "product_review_count",
            "user_mean_rating",
            "product_mean_rating",
            "hours_since_start",
            "user_burstiness_hours",
            "product_burstiness_hours",
        ]
        numeric_features = self.scaler.fit_transform(df[numeric_columns].astype(float))

        text_embeddings = self._encode_texts(df["review_text"].tolist())
        feature_matrix = np.hstack([numeric_features, text_embeddings]).astype(np.float32)
        feature_names = numeric_columns + [f"embedding_{idx}" for idx in range(text_embeddings.shape[1])]
        return FeatureOutput(
            feature_matrix=feature_matrix,
            text_embeddings=text_embeddings.astype(np.float32),
            feature_names=feature_names,
            dataframe=df,
        )

    @staticmethod
    def _uppercase_ratio(text: str) -> float:
        if not text:
            return 0.0
        alpha_chars = [ch for ch in text if ch.isalpha()]
        if not alpha_chars:
            return 0.0
        return sum(ch.isupper() for ch in alpha_chars) / len(alpha_chars)

    @staticmethod
    def _burstiness(df: pd.DataFrame, key: str) -> pd.Series:
        ordered = df[[key, "timestamp"]].sort_values([key, "timestamp"]).copy()
        ordered["time_delta"] = ordered.groupby(key)["timestamp"].diff().dt.total_seconds().div(3600.0)
        ordered["time_delta"] = ordered["time_delta"].fillna(ordered["time_delta"].median())
        return ordered.sort_index()["time_delta"].fillna(0.0)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        try:
            if self.encoder is None:
                self.encoder = SentenceTransformer(
                    self.config.embedding_model_name,
                    local_files_only=True,
                )
            embeddings = self.encoder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)
        except Exception:
            sparse_matrix = self.vectorizer.transform(texts)
            return sparse_matrix.toarray().astype(np.float32)
