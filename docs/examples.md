# Complete Examples

Production-ready code examples for common use cases.

---

## Example 1: Credit Scoring Pipeline (End-to-End)

### Scenario
Bank wants to build credit scorecard for personal loan applications.

```python
from autoriskml import AutoRisk
import pandas as pd

# Initialize AutoRiskML
ar = AutoRisk(
    project='personal_loan_scorecard',
    output_dir='./loan_model_output',
    log_level='INFO'
)

# Load training data
ar.register_source('training', csv='data/loan_applications_2023.csv')

# Configure pipeline for credit scoring
config = {
    'clean': {
        'missing_strategy': 'median',
        'outlier_method': 'iqr',
        'outlier_threshold': 3.0
    },
    'binning': {
        'numeric_method': 'monotonic',  # Industry standard for credit
        'max_bins': 5,
        'min_bin_size': 0.05
    },
    'features': {
        'min_iv': 0.10,  # Only medium+ predictive features
        'max_features': 15,
        'corr_threshold': 0.9
    },
    'models': {
        'logistic': {
            'penalty': 'l2',
            'C': 1.0,
            'class_weight': 'balanced'  # Handle class imbalance
        }
    },
    'scorecard': {
        'pdo': 20,  # FICO-like scale
        'base_score': 600,
        'base_odds': 50.0
    }
}

# Run full pipeline
print("Training credit scorecard...")
result = ar.run(
    source='training',
    target='default_90d',
    config=config,
    scorecard=True,
    explain=True,
    report=True
)

# Review results
print(f"\n✅ Training Complete!")
print(f"Best Model: Logistic Regression")
print(f"AUC: {result.metrics['auc']:.3f}")
print(f"KS: {result.metrics['ks']:.3f}")
print(f"Gini: {result.metrics['gini']:.3f}")

print(f"\nSelected Features ({len(result.selected_features)}):")
for feature, iv in sorted(result.iv_scores.items(), 
                          key=lambda x: x[1], reverse=True)[:10]:
    print(f"  - {feature}: IV = {iv:.4f}")

# Save model for production
result.save('models/loan_scorecard_v1.pkl')
print(f"\n💾 Model saved to: models/loan_scorecard_v1.pkl")
print(f"📊 Report generated: {result.report_html}")
```

**Output:**
```
Training credit scorecard...
[INFO] Stage 1/11: Loading data...
[INFO] Loaded 50,000 records with 45 features
[INFO] Stage 2/11: Profiling data...
[INFO] Stage 3/11: Auto-cleaning...
[INFO] Handled 8 features with missing values
[INFO] Capped outliers in 5 features
[INFO] Stage 4/11: Computing WOE/IV with monotonic binning...
[INFO] Computed WOE/IV for 40 features
[INFO] Stage 5/11: Selecting features...
[INFO] Selected 15 features (min IV=0.10)
[INFO] Stage 6/11: Training models...
[INFO] Training logistic regression...
[INFO] Stage 7/11: Generating scorecard...
[INFO] Stage 8/11: Computing SHAP explanations...
[INFO] Stage 9/11: Computing monitoring metrics...
[INFO] Stage 10/11: Generating report...
[INFO] Stage 11/11: Complete!

✅ Training Complete!
Best Model: Logistic Regression
AUC: 0.823
KS: 0.456
Gini: 0.646

Selected Features (15):
  - debt_to_income: IV = 0.4523
  - credit_score: IV = 0.3812
  - delinquencies_last_2y: IV = 0.3201
  - revolving_utilization: IV = 0.2876
  - ...

💾 Model saved to: models/loan_scorecard_v1.pkl
📊 Report generated: ./loan_model_output/reports/report_20240115.html
```

---

## Example 2: Scoring New Applications

### Scenario
Score new loan applications using trained model.

```python
from autoriskml import AutoRisk
import pandas as pd

# Load trained model
ar = AutoRisk.load('models/loan_scorecard_v1.pkl')

# Load new applications
ar.register_source('new_apps', csv='data/new_applications_jan_2024.csv')

# Score with reason codes
print("Scoring new applications...")
scores = ar.score(
    'new_apps',
    output='with_reasons'  # Include top reason codes
)

# Process scores
df_scores = pd.DataFrame(scores)

print(f"\n📊 Scored {len(df_scores)} applications")
print(f"\nScore Distribution:")
print(df_scores['score'].describe())

print(f"\nRisk Tier Breakdown:")
print(df_scores['risk_tier'].value_counts())

# Example: Review declined applications
declined = df_scores[df_scores['score'] < 650]
print(f"\n❌ Declined Applications: {len(declined)}")

for idx, app in declined.head(3).iterrows():
    print(f"\n--- Application {app['application_id']} ---")
    print(f"Score: {app['score']}")
    print(f"Risk Tier: {app['risk_tier']}")
    print(f"Default Probability: {app['probability']:.2%}")
    print("Top Reasons for Score:")
    for reason in app['reasons'][:4]:
        print(f"  {reason['rank']}. {reason['reason']}")
```

**Output:**
```
Scoring new applications...
[INFO] Loaded 1,250 new applications
[INFO] Applied cleaning spec
[INFO] Applied WOE binning
[INFO] Generated scores and reason codes

📊 Scored 1,250 applications

Score Distribution:
count    1250.000000
mean      623.456789
std        45.234567
min       520.000000
25%       595.000000
50%       625.000000
75%       655.000000
max       735.000000

Risk Tier Breakdown:
Medium Risk        450
Low Risk           380
High Risk          270
Very High Risk      95
Very Low Risk       55

❌ Declined Applications: 365

--- Application APP_2024_001 ---
Score: 585
Risk Tier: High Risk
Default Probability: 8.45%
Top Reasons for Score:
  1. debt_to_income in range (0.45, 0.60] (negative impact: -25 points)
  2. credit_score in range (580, 620] (negative impact: -18 points)
  3. delinquencies_last_2y = 3 (negative impact: -15 points)
  4. revolving_utilization in range (0.85, 1.0] (negative impact: -12 points)
```

---

## Example 3: Production Monitoring

### Scenario
Monitor deployed model for drift and performance degradation.

```python
from autoriskml import AutoRisk
import pandas as pd
from datetime import datetime

# Load model
ar = AutoRisk.load('models/loan_scorecard_v1.pkl')

# Register weekly production data
ar.register_source('prod_week_3', csv='data/production_week_03_2024.csv')

# Monitor for drift
print(f"Monitoring production data (Week 3)...")
monitor_result = ar.monitor(
    source='prod_week_3',
    baseline_source='train'  # Compare to training data
)

# Check results
print(f"\n{'='*60}")
print(f"DRIFT MONITORING REPORT - Week 3, 2024")
print(f"{'='*60}")
print(f"Overall PSI: {monitor_result.overall_psi:.4f}")
print(f"Status: {monitor_result.message}")

if monitor_result.alert:
    print(f"\n⚠️  ALERT: {monitor_result.recommendation}")
    print(f"\nDrifted Features ({len(monitor_result.drifted_features)}):")
    for feature, psi in monitor_result.feature_psi.items():
        if feature in monitor_result.drifted_features:
            print(f"  ❌ {feature}: PSI = {psi:.4f}")
    
    # Trigger retraining workflow
    print("\n🔄 Triggering model retraining...")
    # trigger_retraining_pipeline()
else:
    print("\n✅ No significant drift detected")
    print(f"\nFeature PSI Summary:")
    for feature, psi in sorted(monitor_result.feature_psi.items(), 
                               key=lambda x: x[1], reverse=True)[:5]:
        status = "⚠️" if psi >= 0.1 else "✅"
        print(f"  {status} {feature}: PSI = {psi:.4f}")

# Track performance over time
print(f"\n{'='*60}")
print(f"PERFORMANCE TRACKING")
print(f"{'='*60}")

# Load actual outcomes
prod_data = pd.read_csv('data/production_week_03_2024.csv')
actual_defaults = prod_data['actual_default']
predicted_probs = [s['probability'] for s in ar.score('prod_week_3')]

from autoriskml.metrics import compute_auc, compute_ks_statistic

weekly_auc = compute_auc(actual_defaults, predicted_probs)
weekly_ks, threshold = compute_ks_statistic(actual_defaults, predicted_probs)

print(f"AUC (Week 3): {weekly_auc:.3f}")
print(f"KS (Week 3): {weekly_ks:.3f}")

# Compare to baseline
baseline_auc = 0.823
if weekly_auc < baseline_auc - 0.05:
    print(f"\n⚠️  WARNING: AUC dropped by {baseline_auc - weekly_auc:.3f}")
    print("Consider retraining model!")
else:
    print(f"\n✅ AUC stable (baseline: {baseline_auc:.3f})")
```

**Output (Stable):**
```
Monitoring production data (Week 3)...

============================================================
DRIFT MONITORING REPORT - Week 3, 2024
============================================================
Overall PSI: 0.0782
Status: No significant population shift detected

✅ No significant drift detected

Feature PSI Summary:
  ✅ debt_to_income: PSI = 0.0856
  ✅ credit_score: PSI = 0.0723
  ✅ revolving_utilization: PSI = 0.0689
  ✅ delinquencies_last_2y: PSI = 0.0512
  ✅ loan_amount: PSI = 0.0445

============================================================
PERFORMANCE TRACKING
============================================================
AUC (Week 3): 0.818
KS (Week 3): 0.442

✅ AUC stable (baseline: 0.823)
```

**Output (Drift Detected):**
```
============================================================
DRIFT MONITORING REPORT - Week 7, 2024
============================================================
Overall PSI: 0.2435
Status: Significant population shift detected

⚠️  ALERT: Significant drift detected. Immediate model retraining recommended.

Drifted Features (4):
  ❌ credit_score: PSI = 0.3521
  ❌ debt_to_income: PSI = 0.2876
  ❌ revolving_utilization: PSI = 0.2134
  ❌ income: PSI = 0.2012

🔄 Triggering model retraining...
```

---

## Example 4: Fraud Detection

### Scenario
Real-time fraud detection for e-commerce transactions.

```python
from autoriskml import AutoRisk
import pandas as pd

# Initialize for fraud detection
ar = AutoRisk(
    project='fraud_detection',
    mode='risk'  # 'risk' mode for classification
)

# Register training data
ar.register_source('fraud_train', csv='data/transactions_2023.csv')

# Configure for fraud detection
config = {
    'clean': {
        'missing_strategy': 'mode',  # Categorical features common in fraud
        'outlier_method': None  # Don't remove outliers (fraud is outlier!)
    },
    'binning': {
        'numeric_method': 'quantile',  # Quantile works well for fraud
        'max_bins': 10  # More granularity for fraud patterns
    },
    'features': {
        'min_iv': 0.05,  # Lower threshold (fraud signals can be subtle)
        'max_features': 30
    },
    'models': {
        'xgboost': {  # XGBoost excellent for fraud
            'max_depth': 7,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'scale_pos_weight': 100.0  # Very imbalanced (1% fraud rate)
        }
    }
}

# Train model
print("Training fraud detection model...")
result = ar.run(
    source='fraud_train',
    target='is_fraud',
    config=config,
    scorecard=False,  # No scorecard for fraud (use probabilities)
    explain=True,
    report=True
)

print(f"\n✅ Model trained")
print(f"AUC: {result.metrics['auc']:.3f}")
print(f"KS: {result.metrics['ks']:.3f}")

# Real-time scoring function
def score_transaction(transaction_data):
    """
    Score single transaction in real-time
    
    Returns: (is_fraud, probability, reason)
    """
    # Score transaction
    scores = ar.score(
        transaction_data,
        output='with_reasons'
    )
    
    score_result = scores[0]
    prob = score_result['probability']
    
    # Decision thresholds
    if prob >= 0.90:
        decision = "BLOCK"
        reason = "Very high fraud risk"
    elif prob >= 0.70:
        decision = "REVIEW"
        reason = "High fraud risk - manual review required"
    elif prob >= 0.50:
        decision = "CHALLENGE"
        reason = "Moderate risk - request additional verification"
    else:
        decision = "APPROVE"
        reason = "Low fraud risk"
    
    return {
        'decision': decision,
        'probability': prob,
        'reason': reason,
        'top_factors': score_result['reasons'][:3]
    }

# Example: Score new transaction
new_transaction = pd.DataFrame([{
    'transaction_amount': 2500.00,
    'merchant_category': 'electronics',
    'card_present': False,
    'distance_from_home': 1500,
    'transaction_hour': 3,  # 3 AM
    'velocity_last_hour': 5,  # 5 transactions in last hour
    'new_merchant': True,
    'international': True,
    # ... other features
}])

result = score_transaction(new_transaction)

print(f"\n{'='*60}")
print(f"FRAUD DETECTION RESULT")
print(f"{'='*60}")
print(f"Decision: {result['decision']}")
print(f"Fraud Probability: {result['probability']:.1%}")
print(f"Reason: {result['reason']}")
print(f"\nTop Risk Factors:")
for factor in result['top_factors']:
    print(f"  - {factor['reason']}")
```

**Output:**
```
Training fraud detection model...
[INFO] Loaded 500,000 transactions (1.2% fraud rate)
[INFO] Training XGBoost with class balancing...

✅ Model trained
AUC: 0.947
KS: 0.712

============================================================
FRAUD DETECTION RESULT
============================================================
Decision: REVIEW
Fraud Probability: 78.5%
Reason: High fraud risk - manual review required

Top Risk Factors:
  - transaction_hour in range (0, 6) - Late night/early morning
  - velocity_last_hour = 5 - Multiple transactions in short time
  - international = True - International transaction
```

---

## Example 5: Azure Deployment

### Scenario
Deploy trained model to Azure ML for production serving.

```python
from autoriskml import AutoRisk

# Load trained model
ar = AutoRisk.load('models/loan_scorecard_v1.pkl')

# Deploy to Azure ML
print("Deploying model to Azure ML...")

endpoint = ar.deploy(
    provider='azure_ml',
    workspace_name='risk-ml-workspace',
    resource_group='risk-models-rg',
    subscription_id='your-subscription-id',
    compute_target='aks-production-cluster',
    
    # Deployment configuration
    deployment_name='loan-scorecard-v1',
    cpu_cores=2,
    memory_gb=4,
    auth_enabled=True,
    
    # Auto-scaling
    autoscale_enabled=True,
    autoscale_min_replicas=3,
    autoscale_max_replicas=10,
    autoscale_target_utilization=70
)

print(f"\n✅ Model deployed successfully!")
print(f"Scoring URI: {endpoint['scoring_uri']}")
print(f"Swagger URI: {endpoint['swagger_uri']}")
print(f"Auth Key: {endpoint['auth_key'][:20]}...")

# Test endpoint
import requests
import json

def score_via_api(application_data):
    """
    Score application via deployed API
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {endpoint['auth_key']}"
    }
    
    payload = {
        'data': application_data
    }
    
    response = requests.post(
        endpoint['scoring_uri'],
        headers=headers,
        json=payload
    )
    
    return response.json()

# Test with sample application
test_app = {
    'age': 35,
    'income': 75000,
    'debt_to_income': 0.35,
    'credit_score': 720,
    # ... other features
}

result = score_via_api([test_app])
print(f"\n📊 API Test Result:")
print(json.dumps(result, indent=2))
```

**Output:**
```
Deploying model to Azure ML...
[INFO] Registering model in Azure ML...
[INFO] Creating deployment configuration...
[INFO] Deploying to AKS cluster 'aks-production-cluster'...
[INFO] Waiting for deployment (this may take 5-10 minutes)...
[INFO] Deployment complete!

✅ Model deployed successfully!
Scoring URI: https://loan-scorecard-v1.eastus.azurecontainer.io/score
Swagger URI: https://loan-scorecard-v1.eastus.azurecontainer.io/swagger.json
Auth Key: dGhpc2lzYXNlY3JldGt...

📊 API Test Result:
{
  "results": [
    {
      "score": 685,
      "probability": 0.034,
      "risk_tier": "Low Risk",
      "application_id": null
    }
  ],
  "model_version": "1.0",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

---

## Example 6: Batch Processing

### Scenario
Score millions of existing customers monthly.

```python
from autoriskml import AutoRisk
import pandas as pd
from datetime import datetime

# Load model
ar = AutoRisk.load('models/loan_scorecard_v1.pkl')

# Register large dataset (10 million records)
ar.register_source(
    'all_customers',
    parquet='s3://customer-data/customers_jan_2024.parquet',
    aws_access_key_id='...',
    aws_secret_access_key='...'
)

# Batch score with streaming (memory-efficient)
print("Starting batch scoring of 10M customers...")
start_time = datetime.now()

scores = ar.score(
    'all_customers',
    output='scores',  # Minimal output for speed
    chunk_size=100000  # Process 100k at a time
)

# Write scores to database in chunks
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@host:5432/db')

chunk_results = []
for i, score in enumerate(scores):
    chunk_results.append({
        'customer_id': score.get('id'),
        'score': score['score'],
        'probability': score['probability'],
        'risk_tier': score['risk_tier'],
        'score_date': datetime.now()
    })
    
    # Write every 10k records
    if len(chunk_results) >= 10000:
        df_chunk = pd.DataFrame(chunk_results)
        df_chunk.to_sql(
            'customer_scores',
            engine,
            if_exists='append',
            index=False
        )
        chunk_results = []
        
        if (i + 1) % 100000 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed
            print(f"Processed {i+1:,} customers ({rate:.0f} per second)")

# Write remaining records
if chunk_results:
    df_chunk = pd.DataFrame(chunk_results)
    df_chunk.to_sql('customer_scores', engine, if_exists='append', index=False)

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\n✅ Batch scoring complete!")
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"Average rate: {len(scores)/elapsed:.0f} customers/second")
```

**Output:**
```
Starting batch scoring of 10M customers...
Processed 100,000 customers (2,345 per second)
Processed 200,000 customers (2,378 per second)
...
Processed 10,000,000 customers (2,412 per second)

✅ Batch scoring complete!
Total time: 69.2 minutes
Average rate: 2,406 customers/second
```

---

## More Examples

For more examples, see:
- **Jupyter Notebooks**: `examples/notebooks/`
- **Use Case Guides**: `docs/use_cases/`
- **API Reference**: `docs/api_reference.md`

---

## Support

Questions? We're here to help!

- **GitHub Issues**: https://github.com/idrissbado/AutoRiskML/issues
- **Discussions**: https://github.com/idrissbado/AutoRiskML/discussions
- **Email**: idrissbadoolivier@gmail.com
