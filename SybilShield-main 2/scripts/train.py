from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sybilshield.config import ArtifactConfig, DataConfig, TrainConfig
from sybilshield.training import TrainingPipeline
from sybilshield.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SybilShield GNN models.")
    parser.add_argument("--model", choices=["gcn", "gat"], default="gat")
    parser.add_argument("--num-reviews", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compare-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_config = ArtifactConfig()
    primary_result = None
    comparison_rows: list[dict] = []
    models_to_run = [args.model]
    if args.compare_models:
        models_to_run = ["gcn", "gat"]

    for model_name in models_to_run:
        result = TrainingPipeline(
            data_config=DataConfig(num_reviews=args.num_reviews),
            train_config=TrainConfig(model_name=model_name, epochs=args.epochs, device=args.device),
            artifact_config=artifact_config,
        ).run()
        model_artifact_dir = artifact_config.model_dir(model_name)
        ensure_dir(model_artifact_dir)
        _snapshot_artifacts(artifact_config.root_dir, model_artifact_dir)
        comparison_rows.append(
            {
                "model_name": model_name,
                "roc_auc": result.metrics["test"]["roc_auc"],
                "precision": result.metrics["test"]["precision"],
                "recall": result.metrics["test"]["recall"],
                "f1": result.metrics["test"]["f1"],
                "accuracy": result.metrics["test"]["accuracy"],
            }
        )
        if model_name == args.model:
            primary_result = result

    comparison_payload = {
        "generated_from": models_to_run,
        "models": comparison_rows,
    }
    save_json(artifact_config.model_comparison_path, comparison_payload)

    if primary_result is None:
        raise RuntimeError("Primary model training did not complete.")
    _restore_primary_artifacts(artifact_config.model_dir(args.model), artifact_config.root_dir)
    print(json.dumps(primary_result.metrics, indent=2))


def _snapshot_artifacts(root_dir: Path, destination_dir: Path) -> None:
    for filename in [
        "reviews_with_predictions.csv",
        "metrics.json",
        "history.csv",
        "graph.pkl",
        "checkpoint.pt",
        "run_config.json",
        "roc_curve.png",
        "confusion_matrix.png",
        "fraud_cluster.png",
        "training_history.png",
    ]:
        source = root_dir / filename
        if source.exists():
            shutil.copy2(source, destination_dir / filename)


def _restore_primary_artifacts(source_dir: Path, destination_dir: Path) -> None:
    for filename in [
        "reviews_with_predictions.csv",
        "metrics.json",
        "history.csv",
        "graph.pkl",
        "checkpoint.pt",
        "run_config.json",
        "roc_curve.png",
        "confusion_matrix.png",
        "fraud_cluster.png",
        "training_history.png",
    ]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, destination_dir / filename)


if __name__ == "__main__":
    main()
