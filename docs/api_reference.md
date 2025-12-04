# API Reference

Complete reference for AutoRiskML's public API.

---

## AutoRisk Class

The main entry point for AutoRiskML.

### Constructor

```python
AutoRisk(
    project: str = 'autorisk_project',
    output_dir: str = './autorisk_output',
    log_level: str = 'INFO',
    mode: str = 'risk'
)
```

**Parameters:**
- `project`: Project name for organizing artifacts
- `output_dir`: Directory for outputs (models, reports, etc.)
- `log_level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `mode`: Operation mode (`'risk'` for credit/fraud, `'trading'` for backtesting)

**Example:**
```python
from autoriskml import AutoRisk

ar = AutoRisk(
    project='my_credit_model',
    output_dir='./outputs',
    log_level='INFO'
)
```

---

### register_source()

Register a data source for the pipeline.

```python
ar.register_source(
    name: str,
    csv: str = None,
    parquet: str = None,
    sql_query: str = None,
    s3: str = None,
    **kwargs
)
```

**Parameters:**
- `name`: Unique identifier for this source
- `csv`: Path to CSV file
- `parquet`: Path to Parquet file
- `sql_query`: SQL query with connection details in kwargs
- `s3`: S3 URI (e.g., `s3://bucket/path/data.csv`)
- `**kwargs`: Additional parameters:
  - `connection_string`: For SQL sources
  - `aws_access_key_id`, `aws_secret_access_key`: For S3
  - `kafka_topic`, `kafka_bootstrap_servers`: For Kafka

**Example:**
```python
# CSV source
ar.register_source('training_data', csv='data/train.csv')

# SQL source
ar.register_source(
    'prod_data',
    sql_query='SELECT * FROM loans WHERE status = "approved"',
    connection_string='postgresql://user:pass@host:5432/db'
)

# S3 source
ar.register_source(
    's3_data',
    s3='s3://my-bucket/data/loans.csv',
    aws_access_key_id='...',
    aws_secret_access_key='...'
)
```

---

### run()

Execute the full automated pipeline.

```python
result = ar.run(
    source: str,
    target: str,
    validation_source: str = None,
    config: dict = None,
    clean: bool = True,
    binning: bool = True,
    train: bool = True,
    scorecard: bool = True,
    explain: bool = False,
    monitor: bool = True,
    report: bool = True,
    deploy: bool = False,
    deploy_provider: str = 'azure_ml',
    deploy_config: dict = None
)
```

**Parameters:**
- `source`: Name of registered data source
- `target`: Target column name
- `validation_source`: Optional validation/test dataset
- `config`: Pipeline configuration dictionary
- `clean`: Enable auto-cleaning
- `binning`: Enable WOE/IV binning
- `train`: Train models
- `scorecard`: Generate credit scorecard
- `explain`: Generate SHAP explanations (requires shap package)
- `monitor`: Compute PSI monitoring metrics
- `report`: Generate HTML/PDF report
- `deploy`: Deploy model to cloud
- `deploy_provider`: Deployment platform (`'azure_ml'`, `'aws_sagemaker'`, `'gcp_vertex'`)
- `deploy_config`: Deployment configuration

**Configuration Dictionary:**
```python
config = {
    'clean': {
        'missing_strategy': 'median',  # 'drop', 'median', 'mean', 'mode', 'constant'
        'outlier_method': 'iqr',       # 'iqr', 'zscore', None
        'outlier_threshold': 3.0
    },
    'binning': {
        'numeric_method': 'monotonic',  # 'quantile', 'equal_width', 'monotonic'
        'max_bins': 5,
        'min_bin_size': 0.05
    },
    'features': {
        'min_iv': 0.02,     # Minimum Information Value
        'max_features': 20,  # Maximum number of features
        'corr_threshold': 0.9  # Correlation threshold
    },
    'models': {
        'logistic': {'penalty': 'l2', 'C': 1.0},
        'xgboost': {'max_depth': 5, 'n_estimators': 100},
        'lightgbm': {'num_leaves': 31, 'learning_rate': 0.05}
    },
    'scorecard': {
        'pdo': 20,         # Points to Double Odds
        'base_score': 600,  # Base score
        'base_odds': 50.0   # Base odds ratio
    }
}
```

**Returns:**
`RunResult` object with:
- `profile`: Data profiling report
- `clean_spec`: Cleaning specifications applied
- `binning_spec`: Binning specifications
- `woe_tables`: WOE tables per feature
- `iv_scores`: Information Value scores
- `selected_features`: Selected features
- `best_model`: Best performing model
- `metrics`: Performance metrics (AUC, KS, Gini, etc.)
- `scorecard`: Credit scorecard
- `explanations`: SHAP explanations (if enabled)
- `psi`: PSI values per feature
- `report_html`: Path to HTML report
- `endpoint`: Deployment endpoint (if deployed)

**Example:**
```python
result = ar.run(
    source='training_data',
    target='default',
    config={
        'binning': {'numeric_method': 'monotonic', 'max_bins': 5},
        'features': {'min_iv': 0.02},
        'scorecard': {'pdo': 20, 'base_score': 600}
    },
    explain=True,
    report=True
)

print(f"Best Model AUC: {result.metrics['auc']:.3f}")
print(f"KS Statistic: {result.metrics['ks']:.3f}")
```

---

### score()

Score new data with trained model.

```python
scores = ar.score(
    data: str,  # Path to data or registered source name
    output: str = 'scores',  # 'scores', 'with_reasons', 'full'
    chunk_size: int = None   # For streaming large datasets
)
```

**Parameters:**
- `data`: Path to data file or registered source name
- `output`: Output level:
  - `'scores'`: Probabilities and scores only
  - `'with_reasons'`: Include top reason codes
  - `'full'`: Complete breakdown with all features
- `chunk_size`: Process in chunks for memory efficiency

**Returns:**
List of dictionaries with:
- `score`: Credit score
- `probability`: Default probability
- `risk_tier`: Risk category
- `reasons`: Top reason codes (if `output='with_reasons'` or `'full'`)
- `breakdown`: Full points breakdown (if `output='full'`)

**Example:**
```python
# Score new customers
scores = ar.score(
    'new_customers.csv',
    output='with_reasons'
)

for customer_score in scores:
    print(f"Score: {customer_score['score']}")
    print(f"Risk: {customer_score['risk_tier']}")
    print("Top Reasons:")
    for reason in customer_score['reasons']:
        print(f"  - {reason['reason']}")
```

---

### monitor()

Monitor production data for drift.

```python
monitor_result = ar.monitor(
    source: str,
    current_data: str = None,
    baseline_source: str = 'train'
)
```

**Parameters:**
- `source`: Name of registered current data source
- `current_data`: Direct path to current data (alternative to source)
- `baseline_source`: Baseline for comparison (`'train'`, `'validation'`, or source name)

**Returns:**
`MonitorResult` object with:
- `overall_psi`: Overall PSI across all features
- `feature_psi`: Dictionary of PSI per feature
- `drifted_features`: List of features with significant drift
- `alert`: Boolean flag for drift alert
- `message`: Human-readable summary
- `recommendation`: Action recommendation

**Example:**
```python
monitor_result = ar.monitor(
    source='production_data',
    baseline_source='train'
)

if monitor_result.alert:
    print(f"⚠️ DRIFT ALERT: {monitor_result.message}")
    print(f"Drifted Features: {monitor_result.drifted_features}")
    print(f"Recommendation: {monitor_result.recommendation}")
else:
    print("✅ No significant drift detected")
```

---

### deploy()

Deploy trained model to cloud platform.

```python
endpoint = ar.deploy(
    provider: str = 'azure_ml',
    **kwargs
)
```

**Parameters:**
- `provider`: Platform (`'azure_ml'`, `'aws_sagemaker'`, `'gcp_vertex'`, `'kubernetes'`)
- `**kwargs`: Provider-specific configuration:
  - **Azure ML**: `workspace_name`, `resource_group`, `subscription_id`, `compute_target`
  - **AWS SageMaker**: `role_arn`, `instance_type`, `instance_count`
  - **GCP Vertex AI**: `project_id`, `region`, `endpoint_name`
  - **Kubernetes**: `namespace`, `image`, `replicas`

**Returns:**
Dictionary with:
- `scoring_uri`: Endpoint URL for scoring
- `auth_key`: Authentication key
- `swagger_uri`: API documentation URL
- `deployment_id`: Deployment identifier

**Example:**
```python
# Deploy to Azure ML
endpoint = ar.deploy(
    provider='azure_ml',
    workspace_name='my-ml-workspace',
    resource_group='my-rg',
    subscription_id='...',
    compute_target='aks-cluster'
)

print(f"Model deployed: {endpoint['scoring_uri']}")
```

---

## Binning Module

### compute_woe_iv()

Compute Weight of Evidence and Information Value.

```python
from autoriskml.binning import compute_woe_iv

iv, woe_table, bins = compute_woe_iv(
    values: List,
    target: List[int],
    n_bins: int = 5,
    method: str = 'quantile'
)
```

**Parameters:**
- `values`: Feature values
- `target`: Binary target (0/1)
- `n_bins`: Number of bins
- `method`: Binning method (`'quantile'`, `'equal_width'`, `'monotonic'`)

**Returns:**
- `iv`: Information Value
- `woe_table`: List of tuples `(bin, n_good, n_bad, good_pct, bad_pct, woe)`
- `bins`: Bin edges or categories

**Example:**
```python
iv, woe_table, bins = compute_woe_iv(
    values=[25, 30, 35, 40, 45, 50, 55, 60],
    target=[0, 0, 0, 1, 0, 1, 1, 1],
    n_bins=3,
    method='monotonic'
)

print(f"Information Value: {iv:.4f}")
for bin_info in woe_table:
    print(f"Bin: {bin_info[0]}, WOE: {bin_info[5]:.4f}")
```

---

### compute_psi()

Compute Population Stability Index.

```python
from autoriskml.binning import compute_psi

psi = compute_psi(
    baseline_values: List,
    current_values: List,
    bins: Optional[List] = None
)
```

**Parameters:**
- `baseline_values`: Baseline distribution (training)
- `current_values`: Current distribution (production)
- `bins`: Optional bin edges (computed if not provided)

**Returns:**
- `psi`: PSI value

**Interpretation:**
- PSI < 0.1: Stable
- 0.1 ≤ PSI < 0.2: Moderate drift
- PSI ≥ 0.2: Significant drift (retrain!)

**Example:**
```python
psi = compute_psi(
    baseline_values=[1, 2, 3, 4, 5] * 100,
    current_values=[1, 1, 2, 3, 4] * 100
)

if psi >= 0.2:
    print("⚠️ Significant drift - retrain recommended")
```

---

## Metrics Module

### compute_auc()

Compute Area Under ROC Curve.

```python
from autoriskml.metrics import compute_auc

auc = compute_auc(
    y_true: List[int],
    y_pred_proba: List[float]
)
```

---

### compute_ks_statistic()

Compute Kolmogorov-Smirnov statistic.

```python
from autoriskml.metrics import compute_ks_statistic

ks, threshold = compute_ks_statistic(
    y_true: List[int],
    y_pred_proba: List[float]
)
```

**Returns:**
- `ks`: KS statistic (separation between good/bad)
- `threshold`: Optimal probability threshold

---

### compute_gini()

Compute Gini coefficient.

```python
from autoriskml.metrics import compute_gini

gini = compute_gini(auc: float)
```

**Formula:** `Gini = 2 * AUC - 1`

---

## Scoring Module

### generate_scorecard()

Generate credit scorecard from logistic regression.

```python
from autoriskml.scoring import generate_scorecard

scorecard = generate_scorecard(
    model_coef: Dict[str, float],
    woe_tables: Dict[str, List[Tuple]],
    base_score: int = 600,
    pdo: int = 20,
    base_odds: float = 50.0
)
```

**Parameters:**
- `model_coef`: Logistic regression coefficients
- `woe_tables`: WOE tables per feature
- `base_score`: Base score (e.g., 600 for FICO-like)
- `pdo`: Points to Double Odds (e.g., 20)
- `base_odds`: Base odds ratio (e.g., 50:1 good:bad)

**Returns:**
Scorecard dictionary with points tables per feature.

---

### score_with_scorecard()

Score single observation with scorecard.

```python
from autoriskml.scoring import score_with_scorecard

result = score_with_scorecard(
    data: Dict[str, Any],
    scorecard: Dict,
    woe_tables: Dict
)
```

**Returns:**
- `score`: Credit score
- `probability`: Default probability
- `risk_tier`: Risk category
- `points_breakdown`: Points per feature

---

### explain_score()

Generate reason codes for a score.

```python
from autoriskml.scoring import explain_score

reasons = explain_score(
    score_result: Dict,
    top_n: int = 5
)
```

**Returns:**
List of top contributing factors with impact and points.

---

## Result Objects

### RunResult

Returned by `ar.run()`:

**Attributes:**
- `profile`: Data profiling results
- `woe_tables`: WOE tables per feature
- `iv_scores`: Information Values
- `best_model`: Best model name and object
- `metrics`: Performance metrics dictionary
- `scorecard`: Credit scorecard
- `psi`: PSI per feature
- `report_html`: Path to report

**Methods:**
- `save(path)`: Save results to disk
- `load(path)`: Load from disk

---

### MonitorResult

Returned by `ar.monitor()`:

**Attributes:**
- `overall_psi`: Overall PSI
- `feature_psi`: PSI per feature
- `drifted_features`: Features with drift
- `alert`: Boolean drift alert
- `message`: Summary message
- `recommendation`: Action recommendation

---

## Configuration Reference

### Clean Config

```python
clean_config = {
    'missing_strategy': 'median',  # 'drop', 'median', 'mean', 'mode', 'constant'
    'missing_constant': 0,         # Value if strategy='constant'
    'outlier_method': 'iqr',       # 'iqr', 'zscore', None
    'outlier_threshold': 3.0,      # IQR multiplier or z-score
    'type_coercion': True          # Auto-convert types
}
```

---

### Binning Config

```python
binning_config = {
    'numeric_method': 'monotonic',  # 'quantile', 'equal_width', 'monotonic'
    'max_bins': 5,
    'min_bin_size': 0.05,  # Minimum 5% per bin
    'categorical_max': 10   # Max categories (others grouped)
}
```

---

### Feature Config

```python
features_config = {
    'min_iv': 0.02,          # Minimum IV threshold
    'max_features': 20,      # Maximum features to select
    'corr_threshold': 0.9,   # Remove highly correlated
    'selection_method': 'iv' # 'iv', 'mutual_info', 'chi2'
}
```

---

### Model Config

```python
models_config = {
    'logistic': {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs'
    },
    'xgboost': {
        'max_depth': 5,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'scale_pos_weight': 1.0
    },
    'lightgbm': {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 100
    }
}
```

---

### Scorecard Config

```python
scorecard_config = {
    'pdo': 20,              # Points to Double Odds
    'base_score': 600,      # Base score (midpoint)
    'base_odds': 50.0,      # Base odds ratio
    'round_points': True    # Round points to integers
}
```

---

## Coming Soon

The following modules are planned for future releases:

- **Connectors**: Advanced connectors (Snowflake, BigQuery, Databricks)
- **Profiling**: Enhanced profiling with recommendations
- **Cleaning**: ML-based imputation
- **Models**: Neural networks, ensemble methods
- **Explainability**: LIME integration, custom explainers
- **Monitoring**: Real-time alerting, dashboards
- **Export**: PMML, CoreML formats
- **Deployment**: Additional platforms (GCP, on-premise)

---

## Support

- **PyPI**: https://pypi.org/project/autoriskml/
- **GitHub**: https://github.com/idrissbado/AutoRiskML
- **Issues**: https://github.com/idrissbado/AutoRiskML/issues
- **Email**: idrissbadoolivier@gmail.com
