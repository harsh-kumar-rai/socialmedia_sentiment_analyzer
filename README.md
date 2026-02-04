# Social Media Sentiment Analyzer

A resume-ready showcase project that highlights a sentiment classification pipeline for social media (Twitter) data using classic NLP and machine learning techniques.

## Resume Entry
**Social Media Sentiment Analyzer | Python, NLTK, Scikit-learn**
- Built a sentiment classification pipeline for Twitter data using NLP techniques.
- Implemented text preprocessing, tokenization, and feature extraction workflows.
- Trained a logistic regression model to predict sentiment polarity with strong accuracy.
- Analyzed large datasets to surface insights for social media monitoring.

## Project Highlights
- **NLP preprocessing:** cleaning, normalization, tokenization, and stop-word removal.
- **Feature engineering:** bag-of-words / TF-IDF vectorization for text inputs.
- **Modeling:** logistic regression classifier with evaluation metrics.
- **Insights:** sentiment trends to support social media monitoring and reporting.

## Tech Stack
- Python
- NLTK
- Scikit-learn

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Train a model from the sample dataset:
```bash
python -m sentiment_analyzer.cli train --data data/sample_tweets.csv
```

Train with a validation report:
```bash
python -m sentiment_analyzer.cli train --data data/sample_tweets.csv --evaluate
```

Generate predictions:
```bash
python -m sentiment_analyzer.cli predict --model artifacts/model.bin --data data/sample_tweets.csv
```

## Project Structure
```
sentiment_analyzer/
  cli.py          # Training & prediction CLI
  model.py        # Pipeline + model logic
  preprocess.py   # Text cleaning helpers
data/
  sample_tweets.csv
```

## Future Enhancements
- Add a notebook demo with exploratory data analysis.
- Include a lightweight API or CLI for running predictions.
- Expand to multi-class sentiment and aspect-based analysis.
