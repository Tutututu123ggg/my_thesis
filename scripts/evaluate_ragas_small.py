"""Deprecated thin wrapper.

Evaluation logic now lives in app.evaluation.* so temporary scripts can be
removed later without losing the reasoning/evaluation pipeline.

Prefer:
    python -m app.evaluation.cli
"""

from app.evaluation.cli import main


if __name__ == "__main__":
    main()
