"""Command-line entry point.

    python cli.py explore     # dataset summary + exploratory plots
    python cli.py train       # fit the classifier and report accuracy
    python cli.py cluster     # KMeans segmentation with elbow + PCA plots
    python cli.py gui         # train, then open the prediction window
"""

from __future__ import annotations

import argparse
import logging
import sys

from customer_analytics import clustering, data, modeling, visualization
from customer_analytics.config import DATA_DIR


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="CIH customer behaviour analysis and prediction."
    )
    parser.add_argument("command", choices=["explore", "train", "cluster", "gui"])
    parser.add_argument("--data-dir", default=DATA_DIR, help="directory holding the CSV extracts")
    parser.add_argument("--sample-size", type=int, default=None, help="number of clients to sample")
    parser.add_argument("--clusters", type=int, default=None, help="k for KMeans")
    parser.add_argument("--no-plots", action="store_true", help="compute without rendering figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        merged = data.build_dataset(args.data_dir, args.sample_size)
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"Dataset: {merged.shape[0]} clients, {merged.shape[1]} columns")
    show = not args.no_plots

    if args.command == "explore":
        visualization.overview(merged, show)
        return 0

    if args.command == "cluster":
        segmentation = clustering.segment(merged, args.clusters)
        visualization.elbow(clustering.elbow_scores(segmentation.scaled), show)
        visualization.clusters(segmentation.components, show)
        print("\nCluster profiles:")
        print(segmentation.profiles)
        return 0

    dataset = modeling.build_features(merged)
    trained = modeling.train(dataset)

    print(f"\nRandom Forest accuracy: {trained.accuracy:.4f}")
    print(trained.report)
    print("Top features:")
    print(trained.feature_importances())
    print("\nSample predictions:")
    print(modeling.sample_predictions(trained, dataset))

    if args.command == "gui":
        from customer_analytics.gui import launch

        launch(trained, dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
