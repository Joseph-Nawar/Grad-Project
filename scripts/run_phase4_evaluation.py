"""Run the reproducible Phase 4 evaluation suites."""
from __future__ import annotations
import argparse
from rural_stroke_assist.evaluation.runner import run_evaluation

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "modality", "system", "full"), required=True)
    parser.add_argument("--config", default="config/evaluation/phase4.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing an existing output directory.")
    parser.add_argument("--sklearn-142-python", help="Optional external Python executable for metadata version comparison.")
    args = parser.parse_args()
    output = run_evaluation(args.suite, config_path=args.config, output_dir=args.output_dir, overwrite=args.overwrite, sklearn_142_python=args.sklearn_142_python)
    print(f"Phase 4 evaluation complete: {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
