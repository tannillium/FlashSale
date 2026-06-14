from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sybilshield.config import ArtifactConfig, DataConfig
from sybilshield.data import SyntheticDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic fake review dataset.")
    parser.add_argument("--num-reviews", type=int, default=10000)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = SyntheticDatasetGenerator(DataConfig(num_reviews=args.num_reviews))
    reviews = generator.generate()
    output_path = Path(args.output) if args.output else ArtifactConfig().synthetic_reviews_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(output_path, index=False)
    print(f"Wrote {len(reviews)} reviews to {output_path}")


if __name__ == "__main__":
    main()
