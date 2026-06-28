# Catching the Missed Fraud — Credit Card Fraud Detection Using XGBoost

A live dashboard comparing a Logistic Regression baseline against a tuned XGBoost classifier for
detecting credit card fraud in a severely imbalanced dataset.

**Live dashboard:** https://fraud-detection-xgboost.netlify.app/

## Project context

Group coursework project (MI6228, MSc Business Analytics, Maynooth University). The goal was to
build and evaluate a machine-learning pipeline for detecting fraudulent transactions, explain which
features actually separate fraud from legitimate activity, and frame the results in terms a fraud
operations team would care about — not just headline accuracy.

**Team:** Group 6B — Karishma Yadav, Aadhish, Abhinand, Basant Kumar, Sreejith, Varun
**My contribution:** Feature engineering, XGBoost tuning, SHAP analysis, Gradio deployment

## Key findings

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 67% | 9.2% | 16% | 91.1% |
| XGBoost (default) | 93% | 78.4% | 85.1% | 99.2% |
| **XGBoost (tuned, GridSearchCV)** | **97.2%** | **81.7%** | **87.6%** | **99.9%** |

- The baseline caught only **59 of 644** fraudulent transactions in the test set
- The tuned XGBoost model caught **496** — **437 more** than the baseline, at 97.2% precision
- Card-Not-Present (CNP) proxy transactions show a **3.44% fraud rate** vs **0.02%** for
  Point-of-Sale proxy transactions — over 170x higher risk
- Most influential features (SHAP): `amt_diff_from_avg`, `distance_to_merchant_km`, `CNP_proxy`

## Method

1. Clean and impute the raw transaction data
2. Engineer behavioural features (spending deviation from cardholder average, distance to
   merchant, CNP/POS proxy flags, time-of-day flags) rather than relying on raw fields alone
3. Train a Logistic Regression baseline and a default XGBoost classifier
4. Tune XGBoost via GridSearchCV; apply a 0.7 decision threshold to prioritise precision
5. Use SHAP to explain which features drive predictions

Why this approach: behavioural features keep the model interpretable for a fraud analyst, not just
accurate. Logistic Regression was chosen as the baseline because it's a standard, interpretable
benchmark for binary classification — any added complexity from XGBoost has to clearly beat it.

## Data source

**Kaggle public synthetic credit-card-transaction dataset**, filtered to UK cities (555,719
transactions, 0.4% fraud rate). This is simulated data for research/coursework use — not real
card-network transaction data.

## Tech

Single-file HTML/CSS/vanilla JavaScript. The dot-grid and bar charts are hand-built inline SVG/CSS
(no external chart library). The trained model itself was deployed separately as a Gradio demo
(not included in this repo).

## Limitations

- CNP/POS labels are heuristic proxies, not ground-truth — some misclassification is possible
- Static decision threshold; doesn't adapt per merchant category or in real time
- No real-time data streaming — trained and evaluated on static batch data
- Dataset lacks user demographics and device/browser metadata

## Production roadmap

- Deploy behind a REST API / streaming pipeline for real-time fraud scoring
- Add a feedback loop so analyst-confirmed outcomes retrain the model over time
- Layer in unsupervised anomaly detection for fraud patterns not seen in training
- Tune decision thresholds per merchant category instead of one static cutoff

## Author

**Basant Kumar** — MSc Business Analytics, Maynooth University
