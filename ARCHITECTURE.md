# Architecture: Predictive ML Models

## System Design

```
┌──────────────────────────────────────────────────────────┐
│                  FastAPI Web Service                      │
│              (Health Check, Predictions, Training)       │
└────────────────────┬─────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        │            │            │
   ┌────▼────┐  ┌───▼──────┐  ┌─▼──────────┐
   │Healthcare│  │ Pattern  │  │   Ensemble │
   │Predictor │  │Detector  │  │   Models   │
   │Ensemble  │  │(XGBoost) │  │(Stacking)  │
   └────┬────┘  └───┬──────┘  └─┬──────────┘
        │            │          │
        └────────────┼──────────┘
                     │
        ┌────────────▼──────────┐
        │  Feature Engineering  │
        │  Pipeline             │
        │  - Scaling            │
        │  - Normalization      │
        │  - Temporal features  │
        │  - Technical indicators
        └────────────┬──────────┘
                     │
        ┌────────────▼──────────┐
        │  Data Layer           │
        │  - CSV Loaders        │
        │  - Validators         │
        │  - Train/Val/Test     │
        └───────────────────────┘
```

## Module Organization

### Data Layer (`src/data/`)
- **loader.py**: CSV data loading and validation
- **preprocessor.py**: Feature engineering for healthcare and finance
- **splitter.py**: Train/validation/test splitting with stratification

### Model Layer (`src/models/`)
- **base_model.py**: Abstract base class for all models
- **health_predictor.py**: Ensemble predictor for GAD-7 deterioration
- **pattern_detector.py**: Multi-class pattern classifier (triangle/wedge/flag/other)
- **time_series.py**: LSTM and Prophet forecasting models
- **ensemble.py**: Stacking and advanced ensemble methods

### Evaluation Layer (`src/evaluation/`)
- **metrics.py**: Classification and regression metrics
- **explainability.py**: SHAP and LIME interpretation
- **monitoring.py**: Drift detection and performance monitoring

### API Layer (`src/api/`)
- **routes.py**: FastAPI endpoints for predictions and training
- **models.py**: Pydantic request/response schemas

### Pipelines (`src/pipelines/`)
- **healthcare_pipeline.py**: End-to-end healthcare prediction
- **finance_pipeline.py**: End-to-end pattern detection

## Data Flow

### Training Flow
```
Raw Data (CSV)
    ↓
Data Validation
    ↓
Feature Engineering
    ↓
Data Splitting (70/15/15)
    ↓
Model Training
    ↓
Evaluation & Metrics
    ↓
Model Serialization
```

### Prediction Flow
```
Input Features
    ↓
Feature Preprocessing
    ↓
Model Prediction
    ↓
SHAP Explanation
    ↓
Drift Detection
    ↓
API Response
```

## Key Design Patterns

1. **Abstract Base Model**: All models inherit from `BaseModel` for consistent interface
2. **Ensemble Methods**: Multiple models combined for robustness
3. **Pipeline Architecture**: Modular, composable components
4. **Feature Engineering**: Domain-specific feature creation
5. **Explainability**: SHAP/LIME for model interpretability
6. **Monitoring**: Drift detection and performance tracking

## Technology Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI, Uvicorn |
| ML | scikit-learn, XGBoost, LightGBM |
| Deep Learning | TensorFlow/Keras, PyTorch-ready |
| Time Series | Prophet, statsmodels |
| NLP | NLTK, TextBlob |
| Explainability | SHAP, LIME |
| Data Processing | Pandas, NumPy, SciPy |
| Testing | pytest |
| Containerization | Docker, docker-compose |

## Scalability Considerations

- Models serialized with joblib for fast loading
- Batch prediction support for high-throughput scenarios
- API stateless for horizontal scaling
- Database abstraction for persistence
- Asynchronous training job support
- Model versioning and A/B testing ready
