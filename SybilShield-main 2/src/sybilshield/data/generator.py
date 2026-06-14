from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sybilshield.config import DataConfig


HONEST_OPENERS = [
    "Arrived on time and matched the listing.",
    "Setup was straightforward and the quality felt fair.",
    "Packaging was clean and everything worked out of the box.",
    "It took a little getting used to, but it does the job.",
    "Customer support helped when I had a small issue.",
    "The product feels solid for the price range.",
    "Delivery was smooth and the instructions were easy to follow.",
    "Nothing flashy, but it has been reliable so far.",
]

HONEST_DETAILS = {
    "positive": [
        "I would buy it again if I needed another one.",
        "It met most of my expectations in day-to-day use.",
        "The finish and build quality were better than I expected.",
    ],
    "neutral": [
        "There are a few tradeoffs depending on what you want.",
        "It works, although it did not stand out in any major way.",
        "Overall it was an average experience for me.",
    ],
    "negative": [
        "A couple of small issues kept it from being a better experience.",
        "I expected slightly better durability from the materials.",
        "It is usable, but I probably would not recommend it strongly.",
    ],
}

FRAUD_TEMPLATES = {
    "positive": [
        "Absolutely amazing product highly recommend to everyone",
        "Best purchase ever five stars with zero complaints",
        "Incredible quality and instant results from the first use",
        "Perfect product flawless seller and unbelievable value",
    ],
    "negative": [
        "Worst product ever avoid it at all costs",
        "Total scam and a complete waste of money",
        "Terrible experience from start to finish do not buy",
        "Fake quality and deeply disappointing performance",
    ],
}

TEXT_ATTACK_VARIANTS = [
    "shipping was quick",
    "seller communication was excellent",
    "packaging looked premium",
    "this deserves attention",
    "quality feels unmatched",
    "everyone around me noticed the difference",
]

ATTACK_TYPES = [
    "reviewer_collusion",
    "burst_reviewing",
    "rating_manipulation",
    "text_similarity_attack",
    "coordinated_review_ring",
]


@dataclass(slots=True)
class SyntheticDatasetGenerator:
    config: DataConfig
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate(self) -> pd.DataFrame:
        num_fraud = int(self.config.num_reviews * self.config.fraud_ratio)
        num_honest = self.config.num_reviews - num_fraud

        users = [f"user_{idx:05d}" for idx in range(self.config.num_users)]
        products = [f"product_{idx:05d}" for idx in range(self.config.num_products)]
        user_bias = self._sample_dirichlet(self.config.num_users)
        product_popularity = self._sample_dirichlet(self.config.num_products)

        honest_rows = self._generate_honest_reviews(
            count=num_honest,
            users=users,
            products=products,
            user_bias=user_bias,
            product_popularity=product_popularity,
        )
        fraud_rows = self._generate_fraud_reviews(count=num_fraud, users=users, products=products)

        reviews = pd.DataFrame(honest_rows + fraud_rows)
        reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], utc=False)
        reviews = reviews.sample(frac=1.0, random_state=self.config.random_seed).reset_index(drop=True)
        reviews["review_id"] = [f"review_{idx:05d}" for idx in range(len(reviews))]
        reviews = reviews[
            [
                "review_id",
                "user_id",
                "product_id",
                "rating",
                "review_text",
                "timestamp",
                "is_fraud",
                "ring_id",
                "attack_type",
                "campaign_id",
            ]
        ]
        return reviews

    def _generate_honest_reviews(
        self,
        count: int,
        users: list[str],
        products: list[str],
        user_bias: np.ndarray,
        product_popularity: np.ndarray,
    ) -> list[dict]:
        rows: list[dict] = []
        base_date = pd.Timestamp(self.config.start_date)

        for _ in range(count):
            rating = int(np.clip(np.round(self.rng.normal(3.8, 0.95)), 1, 5))
            sentiment = "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"
            review_text = (
                f"{self.rng.choice(HONEST_OPENERS)} "
                f"{self.rng.choice(HONEST_DETAILS[sentiment])}"
            ).strip()
            timestamp = base_date + pd.Timedelta(
                days=float(self.rng.uniform(0, self.config.days_span)),
                hours=float(self.rng.uniform(0, 24)),
                minutes=float(self.rng.uniform(0, 60)),
            )
            rows.append(
                {
                    "user_id": str(self.rng.choice(users, p=user_bias)),
                    "product_id": str(self.rng.choice(products, p=product_popularity)),
                    "rating": rating,
                    "review_text": review_text,
                    "timestamp": timestamp,
                    "is_fraud": 0,
                    "ring_id": "organic",
                    "attack_type": "organic",
                    "campaign_id": "organic",
                }
            )
        return rows

    def _generate_fraud_reviews(self, count: int, users: list[str], products: list[str]) -> list[dict]:
        rows: list[dict] = []
        base_date = pd.Timestamp(self.config.start_date)
        ring_sizes = self._allocate_reviews(count, self.config.num_rings)
        sybil_pool_size = min(len(users), max(self.config.num_rings * 24, int(self.config.num_users * 0.2)))
        sybil_user_pool = users[:sybil_pool_size]
        targeted_products = self.rng.choice(
            products,
            size=min(len(products), self.config.num_rings * 3),
            replace=False,
        )

        for ring_idx, ring_count in enumerate(ring_sizes):
            ring_id = f"ring_{ring_idx + 1:02d}"
            attack_type = ATTACK_TYPES[ring_idx % len(ATTACK_TYPES)]
            campaign_id = f"{attack_type}_{ring_idx + 1:02d}"
            ring_users = self.rng.choice(
                sybil_user_pool,
                size=int(self.rng.integers(8, 18)),
                replace=False,
            ).tolist()
            product_count = int(self.rng.integers(2, 5))
            target_pool = self.rng.choice(targeted_products, size=product_count, replace=False).tolist()
            attack_start = base_date + pd.Timedelta(days=float(self.rng.uniform(0, self.config.days_span - 2)))
            polarity = int(self.rng.choice([0, 1], p=[0.25, 0.75]))

            for _ in range(ring_count):
                rows.append(
                    self._build_fraud_row(
                        ring_id=ring_id,
                        campaign_id=campaign_id,
                        attack_type=attack_type,
                        ring_users=ring_users,
                        target_products=target_pool,
                        attack_start=attack_start,
                        polarity=polarity,
                    )
                )
        return rows

    def _build_fraud_row(
        self,
        ring_id: str,
        campaign_id: str,
        attack_type: str,
        ring_users: list[str],
        target_products: list[str],
        attack_start: pd.Timestamp,
        polarity: int,
    ) -> dict:
        rating = 5 if polarity else 1
        user_id = str(self.rng.choice(ring_users))
        product_id = str(self.rng.choice(target_products))

        if attack_type == "reviewer_collusion":
            timestamp = attack_start + pd.Timedelta(
                hours=float(self.rng.uniform(0, 30)),
                minutes=float(self.rng.uniform(0, 50)),
            )
            review_text = self._fraud_text(polarity, product_id, heavy_similarity=False)
        elif attack_type == "burst_reviewing":
            timestamp = attack_start + pd.Timedelta(
                minutes=float(self.rng.uniform(0, 180)),
            )
            review_text = self._fraud_text(polarity, product_id, heavy_similarity=True)
        elif attack_type == "rating_manipulation":
            timestamp = attack_start + pd.Timedelta(
                hours=float(self.rng.uniform(0, 18)),
                minutes=float(self.rng.uniform(0, 45)),
            )
            rating = 5 if polarity else 1
            review_text = self._fraud_text(polarity, product_id, heavy_similarity=False)
        elif attack_type == "text_similarity_attack":
            timestamp = attack_start + pd.Timedelta(
                hours=float(self.rng.uniform(0, 12)),
                minutes=float(self.rng.uniform(0, 30)),
            )
            review_text = self._fraud_text(polarity, product_id, heavy_similarity=True)
        else:
            timestamp = attack_start + pd.Timedelta(
                hours=float(self.rng.uniform(0, 36)),
                minutes=float(self.rng.uniform(0, 55)),
            )
            review_text = self._fraud_text(polarity, product_id, heavy_similarity=True)

        return {
            "user_id": user_id,
            "product_id": product_id,
            "rating": rating,
            "review_text": review_text,
            "timestamp": timestamp,
            "is_fraud": 1,
            "ring_id": ring_id,
            "attack_type": attack_type,
            "campaign_id": campaign_id,
        }

    def _fraud_text(self, polarity: int, product_id: str, heavy_similarity: bool) -> str:
        template_group = FRAUD_TEMPLATES["positive" if polarity else "negative"]
        base = str(self.rng.choice(template_group))
        tail = str(self.rng.choice(TEXT_ATTACK_VARIANTS))
        product_hint = f"for {product_id.replace('_', ' ')}"
        punctuation = "!" * int(self.rng.integers(1, 4))
        if heavy_similarity:
            return f"{base} {tail} {product_hint}{punctuation}"
        extra = str(
            self.rng.choice(
                [
                    "would recommend to friends",
                    "seller deserves more visibility",
                    "this changed my opinion immediately",
                    "buyers should avoid missing this",
                ]
            )
        )
        return f"{base} {tail} {product_hint} {extra}{punctuation}"

    def _allocate_reviews(self, total_reviews: int, buckets: int) -> list[int]:
        weights = self.rng.dirichlet(np.ones(buckets) * 1.1)
        counts = np.floor(weights * total_reviews).astype(int)
        remainder = total_reviews - int(counts.sum())
        for idx in self.rng.choice(np.arange(buckets), size=remainder, replace=False):
            counts[int(idx)] += 1
        return counts.tolist()

    def _sample_dirichlet(self, size: int) -> np.ndarray:
        concentration = self.rng.uniform(0.6, 1.8, size=size)
        return self.rng.dirichlet(concentration)
