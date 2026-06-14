# Contributing

## Workflow

1. Create a virtual environment and install dependencies with `pip install -r requirements.txt`.
2. Generate or refresh the synthetic dataset with `python scripts/generate_dataset.py`.
3. Train a model with `python scripts/train.py --model gat --num-reviews 10000 --epochs 10 --compare-models`.
4. Launch the dashboard with `streamlit run app.py`.

## Development Guidelines

- Keep imports clean and remove dead code during changes.
- Prefer small, composable modules under `src/sybilshield/`.
- Save new experiment outputs under `artifacts/models/<model_name>/`.
- Update the README when you change workflow, metrics, or artifact names.

## Pull Requests

- Describe the problem, approach, and verification steps.
- Include before/after screenshots for dashboard changes.
- Mention any dataset-generation or training changes that affect reproducibility.

## Verification Checklist

- `python scripts/generate_dataset.py --num-reviews 10000`
- `python scripts/train.py --model gat --num-reviews 10000 --epochs 10 --compare-models`
- `streamlit run app.py --server.headless true`
