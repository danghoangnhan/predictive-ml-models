# System Architecture

## Overview

This POC implements a production-ready ML platform for healthcare and financial predictions with emphasis on explainability, drift detection, and compliance.

## Core Modules

### 1. Data Pipeline (`src/data/`)
- **loader.py**: CSV loading, validation, HIPAA-ready data handling
- **preprocessor.py**: Feature engineering, outlier detection, categorical encoding
- **splitter.py**: Train/val/test splitting, k-fold CV, time-series split

### 2. Models (`src/models/`)
- **health_predictor.py**: XGBoost/LightGBM classifier for GAD-7 deterioration
- **pattern_detector.py**: Random Forest for technical chart pattern recognition
- **time_series.py**: LSTM + statistical forecasting for price prediction
- **ensemble.py**: Weighted voting ensemble combining multiple models

### 3. Training (`src/training/`)
- **trainer.py**: Model training pipeline with cross-validation
- **hyperopt.py**: Optuna-based hyperparameter tuning
- **cross_validator.py**: K-fold and stratified validation

### 4. Evaluation (`src/evaluation/`)
- **metrics.py**: Classification + regression metrics, fairness metrics
- **explainability.py**: SHAP TreeExplainer, LIME, PDP explanations
- **drift_detector.py**: KS test, Population Stability Index monitoring

### 5. Serving (`src/serving/`)
- **predictor.py**: Real-time inference with explanation generation
- **batch_predictor.py**: CSV batch processing with error handling

### 6. API (`src/api/`)
- **routes.py**: FastAPI endpoints for health/stock predictions
- **models.py**: Pydantic request/response validation

### 7. Configuration (`src/config.py`)
- Environment-based configuration (dev/prod/test)
- HIPAA and financial compliance settings
- Model paths, thresholds, monitoring parameters

## Data Flow

```
Input Data (CSV/API Request)
    ↓
Data Loader (validation, handling missing values)
    ↓
Preprocessor (feature engineering, scaling)
    ↓
Model Prediction (XGBoost/LSTM/Pattern Detector)
    ↓
Explainer (SHAP values)
    ↓
Drift Detector (statistical tests)
    ↓
Output (JSON response with confidence + explanations)
```

## Healthcare Prediction Pipeline

1. **Feature Extraction**: GAD-7 score, age, BMI, sleep patterns
2. **Preprocessing**: Outlier removal, categorical encoding, scaling
3. **Model**: XGBoost binary classifier (deterioration yes/no)
4. **Output**: Risk level (safe/warning/critical) + probability
5. **Explainability**: SHAP force plots showing feature contributions
6. **Monitoring**: Drift detection on incoming patient data

## Finance Prediction Pipeline

1. **Data Ingestion**: OHLCV data from CSV or API
2. **Feature Engineering**: Technical indicators (volatility, support/resistance, momentum)
3. **Pattern Detection**: CNN/RF classifier for chart patterns
4. **Forecasting**: LSTM for 5-30 day price prediction with confidence intervals
5. **Drift Monitoring**: Regime detection for changing market conditions

## Deployment Architecture

```
Client
  ↓
FastAPI (uvicorn)
  ├─ Health endpoint (/predict/health)
  ├─ Stock endpoint (/predict/stock)
  ├─ Batch endpoint (/predict/health/batch)
  ├─ Explain endpoint (/explain/health/{id})
  └─ Drift endpoint (/market/drift)
  ↓
Models (pickled .pkl files)
  ├─ health_model.pkl
  ├─ stock_model.pkl
  └─ scaler (StandardScaler)
  ↓
Disk Storage
  ├─ Data (CSV)
  ├─ Logs (audit trails)
  └─ Models (versioned)
```

## Scalability

- **Horizontal**: Multi-worker Uvicorn with load balancer
- **Vertical**: Batch prediction for 10,000+ samples/day
- **Caching**: SHAP result caching to reduce latency
- **Async**: FastAPI async handlers for concurrent requests

## Monitoring & Compliance

### Healthcare (HIPAA)
- Audit logging of all predictions
- Patient data retention policies (2555 days)
- PII masking options
- Fairness metrics tracking

### Finance (SEC Rule 10b5)
- Model drift detection and alerts
- Feature stability monitoring
- Prediction confidence thresholds
- Backtesting and paper trading support

## Testing Coverage

- **Unit**: Preprocessing, model training, prediction logic
- **Integration**: Data pipeline end-to-end
- **Performance**: Latency benchmarks (<200ms target)
- **Fairness**: Demographic parity and equalized odds

## Model Performance Targets

| Model | Metric | Target |
|-------|--------|--------|
| Health | AUC | > 0.85 |
| Health | Latency | < 100ms |
| Stock | MAE | < $3/share |
| Stock | Directional Accuracy | > 55% |
| Stock | Latency | < 200ms |
