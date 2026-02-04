"""Model training and inference utilities."""

from dataclasses import dataclass
from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment_analyzer.preprocess import normalize_text


@dataclass
class SentimentModel:
    """End-to-end sentiment model using TF-IDF + logistic regression."""

    pipeline: Pipeline

    @classmethod
    def train(cls, texts: Iterable[str], labels: Iterable[str]) -> "SentimentModel":
        """Train a new sentiment model."""
        pipeline = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2)),
                ),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
        pipeline.fit(list(texts), list(labels))
        return cls(pipeline=pipeline)

    def predict(self, texts: Iterable[str]) -> List[str]:
        """Predict sentiment labels for input texts."""
        return [str(label) for label in self.pipeline.predict(list(texts))]
