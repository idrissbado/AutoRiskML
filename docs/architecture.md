# AutoRiskML Architecture

Complete system design and module interactions.

---

## System Overview

AutoRiskML is built as a modular, extensible pipeline for automated risk modeling:

```
┌────────────────────────────────────────────────────────────────┐
│                      AutoRisk API Layer                         │
│  (High-level interface: ar.run(), ar.score(), ar.monitor())   │
└───────────────────────┬────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌───────────┐   ┌──────────────┐   ┌─────────────┐
│Connectors │   │  Core Pipeline│   │ Deployment  │
│ Module    │   │    Engine     │   │   Module    │
└───────────┘   └──────────────┘   └─────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────────────┐
        │               │                       │
        ▼               ▼                       ▼
┌─────────────┐  ┌─────────────┐      ┌─────────────┐
│  Profiling  │  │  Cleaning   │      │   Binning   │
│   Module    │  │   Module    │      │   Module    │
│             │  │             │      │ (WOE/IV)    │
└─────────────┘  └─────────────┘      └─────────────┘
        │               │                       │
        └───────────────┼───────────────────────┘
                        │
        ┌───────────────┼───────────────────────┐
        │               │                       │
        ▼               ▼                       ▼
┌─────────────┐  ┌─────────────┐      ┌─────────────┐
│   Models    │  │   Scoring   │      │   Metrics   │
│   Module    │  │   Module    │      │   Module    │
│             │  │ (Scorecard) │      │ (PSI/KS)    │
└─────────────┘  └─────────────┘      └─────────────┘
        │               │                       │
        └───────────────┼───────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐   ┌──────────────┐
│  Explain    │  │ Monitoring  │   │   Export     │
│  Module     │  │   Module    │   │   Module     │
│  (SHAP)     │  │   (Drift)   │   │  (Reports)   │
└─────────────┘  └─────────────┘   └──────────────┘
```

---

## Module Responsibilities

### 1. API Layer (`api.py`)

**Purpose:** High-level user interface

**Key Classes:**
- `AutoRisk`: Main orchestrator
- `RunResult`: Pipeline results container
- `MonitorResult`: Monitoring results container

**Responsibilities:**
- Project management and directory structure
- Source registration and data loading
- Pipeline orchestration
- Result aggregation and persistence

**Dependencies:** ALL modules (orchestrates entire pipeline)

---

### 2. Connectors Module

**Purpose:** Data ingestion from multiple sources

**Components:**
- `csv_reader.py`: CSV file reader
- `sql_connector.py`: SQL database connector
- `s3_connector.py`: AWS S3 connector
- `kafka_consumer.py`: Kafka stream consumer
- `azure_storage.py`: Azure Blob Storage

**Capabilities:**
- Streaming for large datasets
- Schema inference
- Connection pooling
- Retry logic with exponential backoff

**Example:**
```python
from autoriskml.connectors import CSVReader

reader = CSVReader('large_file.csv', chunk_size=10000)
for chunk in reader:
    process(chunk)
```

---

### 3. Profiling Module

**Purpose:** Automated data profiling

**Features:**
- Data type inference
- Missing value analysis
- Distribution statistics (mean, median, std, skew, kurtosis)
- Outlier detection
- Cardinality analysis
- Correlation matrix
- Recommendations (e.g., "High missing rate in feature X - consider dropping")

**Output:**
```python
{
    'feature_name': {
        'type': 'numeric',
        'missing_pct': 0.05,
        'mean': 45.2,
        'std': 12.3,
        'min': 18,
        'max': 75,
        'outliers_pct': 0.02,
        'recommendation': 'Consider capping outliers at 3 IQR'
    }
}
```

---

### 4. Cleaning Module

**Purpose:** Automated data cleaning

**Strategies:**

**Missing Values:**
- Drop rows/columns
- Median/mean/mode imputation
- Constant value fill
- Forward/backward fill
- KNN imputation (future)
- ML-based imputation (future)

**Outliers:**
- IQR method (1.5 * IQR)
- Z-score method (|z| > 3)
- Capping/flooring
- Removal

**Type Coercion:**
- String → numeric
- Date parsing
- Boolean conversion

**Example:**
```python
from autoriskml.cleaning import AutoCleaner

cleaner = AutoCleaner(
    missing_strategy='median',
    outlier_method='iqr'
)
clean_data, spec = cleaner.fit_transform(data)
```

---

### 5. Binning Module ⭐ REVOLUTIONARY

**Purpose:** WOE/IV computation and binning

**Core Functions:**

**`compute_woe_iv()`**
- Automatic numeric vs categorical detection
- Three binning methods:
  1. **Quantile**: Equal frequency bins
  2. **Equal Width**: Equal range bins
  3. **Monotonic**: Advanced credit scoring technique
- WOE computation with smoothing
- IV calculation and interpretation

**Monotonic Binning Algorithm:**
```
1. Start with 20 initial bins
2. Compute bad rate per bin
3. Check monotonicity (increasing or decreasing)
4. While not monotonic:
   a. Find adjacent bins with most similar bad rates
   b. Merge these bins
   c. Recompute bad rates
5. Continue until monotonic or target bins reached
```

**Why Monotonic Binning?**
- Ensures logical risk ordering
- Industry standard for credit scorecards
- Regulatory compliance (explainable risk progression)
- Prevents score inversions

**`compute_psi()`**
- Population Stability Index
- Drift detection for production monitoring
- Automatic alert triggers

**Example:**
```python
from autoriskml.binning import compute_woe_iv

iv, woe_table, bins = compute_woe_iv(
    values=age_values,
    target=default_labels,
    n_bins=5,
    method='monotonic'  # Advanced technique!
)

print(f"IV: {iv:.4f}")
for bin_info in woe_table:
    print(f"Age {bin_info[0]}: WOE = {bin_info[5]:.4f}")
```

---

### 6. Models Module

**Purpose:** Model training and selection

**Supported Algorithms:**
- Logistic Regression (primary for scorecard)
- XGBoost
- LightGBM
- CatBoost (future)
- Neural Networks (future)

**Features:**
- Hyperparameter tuning (GridSearchCV, RandomSearchCV)
- Cross-validation
- Class balancing (SMOTE, class weights)
- Early stopping
- Model comparison and selection

**Model Wrappers:**
```python
from autoriskml.models import LogisticModel, XGBoostModel

# Unified interface
logistic = LogisticModel(C=1.0, penalty='l2')
logistic.fit(X_train, y_train)
proba = logistic.predict_proba(X_test)
```

---

### 7. Scoring Module

**Purpose:** Credit scorecard generation

**Key Components:**

**`generate_scorecard()`**
- Converts logistic regression to scorecard
- PDO (Points to Double Odds) scaling
- Points calculation per WOE bin

**Formula:**
```
Score = base_score + (pdo / ln(2)) * (ln(odds) - ln(base_odds))
Points_i = (pdo / ln(2)) * coefficient_i * WOE_i
```

**Scorecard Structure:**
```python
{
    'base_score': 600,
    'pdo': 20,
    'features': {
        'age': {
            'coefficient': 0.45,
            'points_table': [
                {'bin': '18-25', 'woe': -0.5, 'points': -10},
                {'bin': '26-35', 'woe': 0.0, 'points': 0},
                {'bin': '36-45', 'woe': 0.3, 'points': +7},
                {'bin': '46+', 'woe': 0.6, 'points': +13}
            ]
        }
    }
}
```

**`explain_score()`**
- Reason codes (top factors)
- Adverse action requirements (regulatory)
- Feature contributions

---

### 8. Metrics Module

**Purpose:** Model performance evaluation

**Metrics Implemented:**

**Classification:**
- AUC (Area Under ROC Curve)
- KS Statistic (Kolmogorov-Smirnov)
- Gini Coefficient
- Brier Score (calibration)
- Confusion Matrix (TP, TN, FP, FN)
- Precision, Recall, F1

**Risk-Specific:**
- PSI (Population Stability Index)
- CSI (Characteristic Stability Index)
- Lift Chart
- Cumulative Accuracy Profile

**Example:**
```python
from autoriskml.metrics import compute_ks_statistic, compute_auc

auc = compute_auc(y_true, y_pred_proba)
ks, threshold = compute_ks_statistic(y_true, y_pred_proba)

print(f"AUC: {auc:.3f}")
print(f"KS: {ks:.3f} at threshold {threshold:.2f}")
```

---

### 9. Explainability Module

**Purpose:** Model interpretability

**Techniques:**
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance
- Partial dependence plots

**Global Explanations:**
- Feature importance ranking
- Global SHAP values

**Local Explanations:**
- Individual prediction explanations
- Reason codes for adverse actions

**Example:**
```python
from autoriskml.explain import SHAPExplainer

explainer = SHAPExplainer(model)
shap_values = explainer.explain_global(X_train)
explainer.plot_feature_importance()
```

---

### 10. Monitoring Module

**Purpose:** Production model monitoring

**Components:**

**Drift Detection:**
- PSI per feature
- CSI per feature
- Overall drift score
- Alert triggers

**Performance Monitoring:**
- AUC over time
- KS over time
- Prediction distribution shifts
- Bad rate drift

**Alerting:**
- Email alerts
- Slack/Teams webhooks
- Custom callbacks

**Example:**
```python
from autoriskml.monitoring import DriftMonitor

monitor = DriftMonitor(
    baseline_data=train_data,
    alert_threshold=0.2
)

result = monitor.check_drift(production_data)
if result.alert:
    send_alert(result.message)
```

---

### 11. Export Module

**Purpose:** Model serialization and reporting

**Formats:**
- Joblib (Python)
- Pickle (Python)
- ONNX (cross-platform)
- PMML (future)
- CoreML (future)

**Reports:**
- HTML report (interactive)
- PDF report (static)
- Excel scorecard
- JSON API specs

**Example:**
```python
from autoriskml.export import export_onnx, generate_html_report

# Export model
export_onnx(model, 'model.onnx')

# Generate report
generate_html_report(result, 'report.html')
```

---

### 12. Deployment Module

**Purpose:** Cloud deployment

**Platforms:**

**Azure ML:**
- ACI (Container Instances) for dev/test
- AKS (Kubernetes Service) for production
- Managed endpoints

**AWS SageMaker:**
- Real-time endpoints
- Batch transform
- Multi-model endpoints

**GCP Vertex AI:**
- Prediction endpoints
- Batch predictions

**Kubernetes:**
- Custom deployments
- Helm charts
- Horizontal autoscaling

**Example:**
```python
from autoriskml.deployment import AzureMLDeployer

deployer = AzureMLDeployer(
    workspace_name='my-workspace',
    resource_group='my-rg'
)

endpoint = deployer.deploy(
    model=trained_model,
    compute_target='aks-cluster',
    name='risk-model-v1'
)

print(f"Deployed: {endpoint['scoring_uri']}")
```

---

## Data Flow

### Training Pipeline

```
Raw Data
    │
    ├─► Connectors ──► Load data from CSV/SQL/S3/Kafka
    │
    ├─► Profiling ──► Analyze data quality, types, distributions
    │
    ├─► Cleaning ──► Handle missing values, outliers, types
    │
    ├─► Binning ──► Compute WOE/IV with monotonic binning
    │
    ├─► Feature Selection ──► Select features with IV > threshold
    │
    ├─► Models ──► Train logistic/XGBoost/LightGBM
    │
    ├─► Scoring ──► Generate credit scorecard (PDO scaling)
    │
    ├─► Metrics ──► Evaluate AUC, KS, Gini, PSI
    │
    ├─► Explain ──► SHAP explanations
    │
    ├─► Export ──► Save model, scorecard, reports
    │
    └─► Deploy ──► Deploy to Azure ML/AWS/GCP
```

### Scoring Pipeline

```
New Data
    │
    ├─► Load ──► From file or API
    │
    ├─► Clean ──► Apply saved cleaning spec
    │
    ├─► Bin ──► Apply saved WOE bins
    │
    ├─► Score ──► Apply scorecard for points
    │
    ├─► Explain ──► Generate reason codes
    │
    └─► Output ──► Score, probability, risk tier, reasons
```

### Monitoring Pipeline

```
Production Data
    │
    ├─► Load ──► Stream from Kafka or batch from storage
    │
    ├─► Compute PSI ──► Compare to baseline (train data)
    │
    ├─► Check Thresholds ──► PSI >= 0.2 triggers alert
    │
    ├─► Alert ──► Email/Slack notification
    │
    └─► Log ──► Store drift metrics for trending
```

---

## Extensibility

### Adding Custom Components

**Custom Binning Method:**
```python
from autoriskml.binning import register_binning_method

@register_binning_method('custom')
def my_custom_binning(values, target, n_bins):
    # Your binning logic
    return bins

# Use it
compute_woe_iv(values, target, method='custom')
```

**Custom Metric:**
```python
from autoriskml.metrics import register_metric

@register_metric('my_metric')
def my_custom_metric(y_true, y_pred):
    # Your metric calculation
    return score
```

**Custom Connector:**
```python
from autoriskml.connectors import BaseConnector

class MyConnector(BaseConnector):
    def load(self):
        # Load data from custom source
        return data

ar.register_source('custom', connector=MyConnector(...))
```

---

## Performance Considerations

### Memory Management

**Streaming for Large Datasets:**
```python
ar.register_source('large_data', csv='huge.csv')
ar.run(source='large_data', chunk_size=10000)  # Process in chunks
```

**Dask Integration (Future):**
```python
ar.run(source='data', distributed='dask')
```

### Computation

**Parallel Processing:**
- Cross-validation uses all CPU cores
- Feature selection parallelized
- Scoring supports multi-threading

**GPU Acceleration:**
- XGBoost with `tree_method='gpu_hist'`
- LightGBM with `device='gpu'`

---

## Testing Strategy

### Unit Tests
- Each module tested independently
- Mock data generators
- Edge case coverage

### Integration Tests
- End-to-end pipeline tests
- Multi-source connectors
- Deployment validation

### Performance Tests
- Large dataset benchmarks
- Memory profiling
- Latency measurements

---

## Security

### Data Protection
- Sensitive data encryption at rest
- Secure credential management (Azure Key Vault)
- PII detection and masking

### Model Security
- Model versioning and audit trails
- Access control for endpoints
- API rate limiting

### Compliance
- GDPR compliance features
- Right to explanation (reason codes)
- Audit logging

---

## Future Enhancements

### Planned Features
- AutoML for hyperparameter tuning
- Neural network support
- Time series risk models
- Real-time streaming scoring
- Multi-model ensembles
- Federated learning
- Advanced drift detection (Wasserstein distance)

### Community Contributions
We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Support

- **GitHub Issues**: https://github.com/idrissbado/AutoRiskML/issues
- **Discussions**: https://github.com/idrissbado/AutoRiskML/discussions
- **Email**: idrissbadoolivier@gmail.com
