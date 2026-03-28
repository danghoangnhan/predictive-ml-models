# Predictive ML Models: Healthcare & Finance

A production-ready POC for predictive machine learning models serving healthcare (patient health deterioration prediction) and finance (stock pattern detection) domains with explainability and monitoring.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│           FastAPI Prediction Server                      │
│  (/predict/health, /predict/pattern, /train, /health)   │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
   ┌────▼────┐ ┌──▼──────┐ ┌─▼──────────────┐
   │Healthcare│ │ Finance  │ │ Ensemble       │
   │Predictor │ │Predictor │ │Models          │
   │(GAD-7)   │ │(Patterns)│ │(XGBoost/LGBM)  │
   └────┬────┘ └──┬──────┘ └─┬──────────────┘
        │         │          │
   ┌────▼─────────▼──────────▼────┐
   │  Feature Engineering Pipeline │
   │  (Preprocessing, Scaling)     │
   └────┬─────────────────────────┘
        │
   ┌────▼──────────────────────────┐
   │  Data Loaders & Splitters      │
   │  (Train/Val/Test)              │
   └───────────────────────────────┘
```

## Models & Capabilities

### 1. Healthcare Predictor (GAD-7)
- **Input**: Patient GAD-7 scores, journal text entries, temporal data
- **Output**: Deterioration risk prediction (binary classification)
- **Features**:
  - Time series trend analysis (7-day, 14-day moving averages)
  - NLP feature extraction from journal entries (sentiment, keyword frequency)
  - Temporal features (day-of-week, week-of-year)
- **Model**: Logistic Regression + Random Forest + Neural Network ensemble
- **Metrics**: AUC-ROC, Precision, Recall, F1-Score

### 2. Finance Pattern Detector
- **Input**: Stock OHLCV (Open, High, Low, Close, Volume) data
- **Output**: Chart pattern classification (triangle, wedge, flag, other)
- **Features**:
  - Candlestick pattern recognition
  - Volatility indices (ATR, Bollinger Bands)
  - Volume-weighted metrics
  - Support/Resistance levels
- **Model**: CNN-based or XGBoost pattern classifier
- **Metrics**: Accuracy, Precision, Recall per pattern class

### 3. Time Series Forecasting
- **LSTM**: Deep learning for sequential prediction
- **Prophet**: Facebook's time series framework for trend + seasonality
- Applicable to both domains (patient trajectories, stock prices)

### 4. Ensemble Methods
- **XGBoost**: Gradient boosting for feature importance
- **LightGBM**: Fast, memory-efficient tree-based model
- **Stacking**: Meta-learner combining multiple models
- Feature importance via SHAP values

## Project Structure

```
predictive-ml-models/
├── src/
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration management
│   ├── data/
│   │   ├── loader.py              # Data loading
│   │   ├── preprocessor.py        # Feature engineering
│   │   └── splitter.py            # Train/val/test splitting
│   ├── models/
│   │   ├── base_model.py          # Base model class
│   │   ├── health_predictor.py    # Healthcare models
│   │   ├── pattern_detector.py    # Finance models
│   │   ├── time_series.py         # LSTM & Prophet
│   │   └── ensemble.py            # Ensemble & stacking
│   ├── evaluation/
│   │   ├── metrics.py             # Metrics
│   │   ├── explainability.py      # SHAP/LIME
│   │   └── monitoring.py          # Drift detection
│   ├── pipelines/
│   │   ├── healthcare_pipeline.py # Healthcare workflow
│   │   └── finance_pipeline.py    # Finance workflow
│   └── api/
│       ├── routes.py              # API endpoints
│       └── models.py              # Pydantic models
├── data/sample/
│   ├── health_scores.csv          # Synthetic GAD-7 data
│   └── stock_patterns.csv         # Synthetic patterns
├── tests/
├── scripts/
│   ├── train.py                   # Training script
│   ├── predict.py                 # Prediction script
│   ├── evaluate.py                # Evaluation script
│   └── generate_sample_data.py    # Data generation
├── notebooks/
│   ├── eda_healthcare.ipynb       # Healthcare EDA
│   └── eda_finance.ipynb          # Finance EDA
├── configs/
│   ├── healthcare_config.yaml     # Healthcare config
│   └── finance_config.yaml        # Finance config
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── PROPOSAL.md
└── README.md
```

## Quick Start

### Local Development

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate sample data:
   ```bash
   python scripts/generate_sample_data.py
   ```

4. Train models:
   ```bash
   python scripts/train.py --domain healthcare
   python scripts/train.py --domain finance
   ```

5. Run API:
   ```bash
   python src/main.py
   ```
   API at `http://localhost:8000`

### Docker

```bash
docker-compose up --build
```

## API Endpoints

### POST /predict/health
Predict health deterioration risk.

**Request**:
```json
{
  "patient_id": "P123",
  "gad7_score": 18,
  "journal_text": "Feeling anxious",
  "days_since_last_assessment": 7
}
```

### POST /predict/pattern
Classify stock patterns.

**Request**:
```json
{
  "symbol": "AAPL",
  "ohlcv": [[100, 105, 98, 102, 1000]],
  "pattern_lookback_days": 20
}
```

### GET /health
Service health check.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Explainability

- SHAP values for feature importance
- LIME for local explanations
- Drift detection and monitoring
- Real-time prediction logging

## License

MIT
