"""Command line interface for the sentiment analyzer."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sentiment_analyzer.model import SentimentModel


def read_csv_texts(
    path: Path, text_column: str, label_column: str | None = None
) -> Tuple[List[str], List[str]]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        texts = []
        labels = []
        for row in reader:
            texts.append(row[text_column])
            if label_column:
                labels.append(row[label_column])
    return texts, labels


def write_predictions(path: Path, texts: List[str], predictions: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["text", "prediction"])
        writer.writerows(zip(texts, predictions))


def train_command(args: argparse.Namespace) -> None:
    texts, labels = read_csv_texts(Path(args.data), args.text_column, args.label_column)
    if args.evaluate:
        label_counts = Counter(labels)
        can_stratify = all(count >= 2 for count in label_counts.values())
        if not can_stratify:
            print(
                "Warning: Not enough samples per class to stratify; "
                "using a random split instead."
            )
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=labels if can_stratify else None,
        )
        model = SentimentModel.train(train_texts, train_labels)
        predictions = model.predict(test_texts)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Validation accuracy: {accuracy:.3f}")
        print(classification_report(test_labels, predictions))

    model = SentimentModel.train(texts, labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, output_path)
    print(f"Model saved to {output_path}")


def predict_command(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    pipeline = joblib.load(model_path)
    model = SentimentModel(pipeline=pipeline)
    texts, _ = read_csv_texts(Path(args.data), args.text_column)
    predictions = model.predict(texts)
    write_predictions(Path(args.output), texts, predictions)
    print(f"Predictions saved to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Social media sentiment analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a sentiment model")
    train_parser.add_argument("--data", required=True, help="CSV file with text + labels")
    train_parser.add_argument("--text-column", default="text")
    train_parser.add_argument("--label-column", default="label")
    train_parser.add_argument("--output", default="artifacts/model.bin")
    train_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Print validation metrics using a holdout split.",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser("predict", help="Run predictions")
    predict_parser.add_argument("--model", required=True, help="Path to model file")
    predict_parser.add_argument("--data", required=True, help="CSV file with text")
    predict_parser.add_argument("--text-column", default="text")
    predict_parser.add_argument("--output", default="artifacts/predictions.csv")
    predict_parser.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
