# Persian Text Sentiment Analysis

**Objective:** Classify Persian text into sentiment categories using BERT.

**Dataset:** Sample dataset provided in `data/`. (For full datasets, use [link to Persian dataset])

**Methods:**
- Text preprocessing: tokenization, normalization, stop-word removal
- Model: BERT-base fine-tuned for classification
- Evaluation: Accuracy, F1-score, confusion matrix

**Results:** Example: 91% accuracy and 0.89 F1-score on test set.

**Usage:**
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebook `notebooks/sentiment_analysis.ipynb` for step-by-step preprocessing, training, and evaluation
3. Run `scripts/train_model.py` for full model training
