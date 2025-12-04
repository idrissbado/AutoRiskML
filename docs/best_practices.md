# Best Practices for AutoRiskML

Production-ready guidelines for building robust risk models.

---

## Data Preparation

### 1. Data Quality Checks

**Always profile your data first:**
```python
ar = AutoRisk(project='credit_model')
ar.register_source('data', csv='applications.csv')

# Run profiling only
result = ar.run(
    source='data',
    target='default',
    train=False,  # Skip training to just profile
    report=True
)

# Review profiling report before proceeding
print(result.profile)
```

**Check for:**
- Missing value patterns (MAR vs MCAR vs MNAR)
- Outliers and extreme values
- Class imbalance
- Data leakage (future information in features)
- Duplicate records
- Inconsistent data types

---

### 2. Feature Engineering

**Create domain-specific features:**
```python
# Before AutoRiskML
import pandas as pd

df = pd.read_csv('loans.csv')

# Create risk-relevant features
df['debt_to_income'] = df['monthly_debt'] / df['monthly_income']
df['credit_utilization'] = df['credit_used'] / df['credit_limit']
df['loan_to_value'] = df['loan_amount'] / df['property_value']
df['payment_to_income'] = df['monthly_payment'] / df['monthly_income']

# Save enhanced data
df.to_csv('loans_engineered.csv', index=False)

# Then use AutoRiskML
ar.register_source('data', csv='loans_engineered.csv')
```

**Domain knowledge beats algorithms!**

---

### 3. Handling Class Imbalance

**For imbalanced datasets (e.g., 1% default rate):**

```python
config = {
    'models': {
        'logistic': {
            'class_weight': 'balanced'  # Automatically balance classes
        },
        'xgboost': {
            'scale_pos_weight': 99.0  # For 1% positive class (99:1 ratio)
        }
    }
}

result = ar.run(
    source='data',
    target='default',
    config=config
)
```

**Alternative: Use stratified sampling:**
```python
from sklearn.model_selection import train_test_split

# Ensure balanced train/test split
train, test = train_test_split(
    df,
    test_size=0.2,
    stratify=df['default'],  # Preserve class ratio
    random_state=42
)
```

---

## Binning Strategy

### 1. Choose the Right Method

**Quantile Binning:**
- Use for: Uniform sample distribution
- Pros: Equal-sized bins, stable
- Cons: May not respect natural boundaries

**Equal Width Binning:**
- Use for: Normally distributed features
- Pros: Simple, interpretable ranges
- Cons: May create empty bins

**Monotonic Binning:** ⭐ RECOMMENDED for Credit Scoring
- Use for: Credit scorecards, regulatory compliance
- Pros: Logical risk progression, explainable
- Cons: Slightly more computation

```python
config = {
    'binning': {
        'numeric_method': 'monotonic',  # Best for credit risk!
        'max_bins': 5,
        'min_bin_size': 0.05  # Each bin >= 5% of data
    }
}
```

---

### 2. Optimal Number of Bins

**Rules of thumb:**
- **5 bins**: Standard for most features (good balance)
- **3-4 bins**: For highly predictive features (stronger WOE)
- **6-8 bins**: For complex non-linear relationships

**Too few bins:** Loss of predictive power  
**Too many bins:** Overfitting, unstable in production

```python
# Feature-specific binning
config = {
    'binning': {
        'numeric_method': 'monotonic',
        'max_bins': {
            'age': 5,
            'income': 7,
            'credit_score': 4
        }
    }
}
```

---

### 3. Handling Special Values

**Missing values:**
```python
# Create separate "Missing" bin
config = {
    'clean': {
        'missing_strategy': 'constant',
        'missing_constant': -999  # Special missing indicator
    },
    'binning': {
        'treat_missing_separately': True  # Separate bin for -999
    }
}
```

---

## Feature Selection

### 1. Information Value Thresholds

**IV Guidelines:**
```
< 0.02: Not useful (drop)
0.02-0.1: Weak (consider dropping)
0.1-0.3: Medium (keep)
0.3-0.5: Strong (keep)
>= 0.5: Suspicious (check for data leakage!)
```

**Conservative approach:**
```python
config = {
    'features': {
        'min_iv': 0.10,  # Only medium+ features
        'max_features': 15  # Limit for scorecard simplicity
    }
}
```

**Aggressive approach (more features):**
```python
config = {
    'features': {
        'min_iv': 0.02,  # Include weak features
        'max_features': 30
    }
}
```

---

### 2. Multicollinearity

**Remove highly correlated features:**
```python
config = {
    'features': {
        'corr_threshold': 0.9,  # Remove if correlation > 0.9
        'corr_method': 'iv'  # Keep feature with higher IV
    }
}
```

**Why?**
- Correlated features add no new information
- Inflate model complexity
- Make scorecards confusing

---

## Model Selection

### 1. Logistic Regression for Scorecards

**Always use logistic regression for credit scorecards:**
```python
config = {
    'models': {
        'logistic': {
            'penalty': 'l2',  # Ridge regularization
            'C': 1.0,  # Regularization strength (lower = more regularization)
            'solver': 'lbfgs',
            'max_iter': 1000
        }
    }
}

result = ar.run(..., config=config, scorecard=True)
```

**Why logistic regression?**
- Linear in log-odds (required for scorecard)
- Interpretable coefficients
- Regulatory acceptance
- Easy to explain

---

### 2. Tree Models for Maximum Performance

**Use XGBoost/LightGBM for highest AUC:**
```python
config = {
    'models': {
        'xgboost': {
            'max_depth': 5,  # Deeper = more complex
            'n_estimators': 200,
            'learning_rate': 0.05,  # Lower = more stable
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10.0  # For imbalanced data
        }
    }
}
```

**Trade-offs:**
- Higher AUC, but no scorecard
- Less interpretable
- Harder to explain to regulators

---

### 3. Ensemble Strategy

**Use multiple models:**
```python
config = {
    'models': {
        'logistic': {...},  # For scorecard
        'xgboost': {...},  # For maximum AUC
        'lightgbm': {...}  # Another strong performer
    }
}

result = ar.run(..., config=config)

# AutoRiskML selects best model by AUC
print(f"Best model: {result.best_model['name']}")
print(f"AUC: {result.metrics['auc']:.3f}")
```

---

## Scorecard Design

### 1. PDO Selection

**Common PDO values:**
```python
# Conservative (small point changes)
config = {'scorecard': {'pdo': 10}}  # 10 points to double odds

# Standard (FICO-like)
config = {'scorecard': {'pdo': 20}}  # 20 points to double odds

# Aggressive (large point swings)
config = {'scorecard': {'pdo': 50}}  # 50 points to double odds
```

**Recommendation:** Use PDO=20 (industry standard)

---

### 2. Base Score Selection

**Common scales:**
```python
# FICO-like scale (300-850)
config = {'scorecard': {'base_score': 600, 'pdo': 20}}

# Simpler scale (0-1000)
config = {'scorecard': {'base_score': 500, 'pdo': 50}}

# Custom scale (1-100)
config = {'scorecard': {'base_score': 50, 'pdo': 10}}
```

**Recommendation:** Use base_score=600 (familiar to users)

---

### 3. Risk Tiers

**Define cutoffs based on business needs:**
```python
# After scoring
scores = ar.score('new_customers.csv', output='full')

# Custom risk tiers
def assign_tier(score):
    if score >= 700:
        return "Approve"
    elif score >= 650:
        return "Manual Review"
    else:
        return "Decline"

for s in scores:
    tier = assign_tier(s['score'])
    print(f"Score: {s['score']}, Decision: {tier}")
```

---

## Production Monitoring

### 1. PSI Monitoring

**Set appropriate thresholds:**
```python
# Monitor production data weekly
monitor_result = ar.monitor(
    source='production_week_45',
    baseline_source='train'
)

if monitor_result.overall_psi >= 0.2:
    # CRITICAL: Retrain immediately
    send_alert("Model drift detected! PSI = {:.3f}".format(
        monitor_result.overall_psi
    ))
    trigger_retraining()

elif monitor_result.overall_psi >= 0.1:
    # WARNING: Monitor closely
    log_warning("Moderate drift detected. Continue monitoring.")
```

**PSI Interpretation:**
- PSI < 0.1: **Stable** (no action)
- 0.1 ≤ PSI < 0.2: **Moderate drift** (monitor weekly)
- PSI ≥ 0.2: **Critical drift** (retrain immediately!)

---

### 2. Performance Monitoring

**Track AUC over time:**
```python
import pandas as pd

# Weekly performance tracking
performance_log = []

for week in range(1, 53):
    week_data = load_week_data(week)
    predictions = ar.score(week_data)
    
    actual = week_data['actual_default']
    predicted = [p['probability'] for p in predictions]
    
    from autoriskml.metrics import compute_auc
    weekly_auc = compute_auc(actual, predicted)
    
    performance_log.append({
        'week': week,
        'auc': weekly_auc,
        'n_samples': len(week_data)
    })

df_perf = pd.DataFrame(performance_log)

# Alert if AUC drops
if df_perf['auc'].iloc[-1] < 0.70:
    send_alert("Model AUC dropped below 0.70!")
```

---

### 3. Data Quality Monitoring

**Monitor input data quality:**
```python
def check_data_quality(new_data):
    issues = []
    
    # Missing values
    missing_pct = new_data.isnull().mean()
    if any(missing_pct > 0.5):
        issues.append("High missing values detected")
    
    # Outliers
    for col in numeric_columns:
        q1, q3 = new_data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((new_data[col] < q1 - 3*iqr) | 
                    (new_data[col] > q3 + 3*iqr)).sum()
        if outliers / len(new_data) > 0.1:
            issues.append(f"High outliers in {col}")
    
    # Range checks
    if (new_data['age'] < 0).any() or (new_data['age'] > 120).any():
        issues.append("Invalid age values")
    
    return issues

# Before scoring
issues = check_data_quality(production_data)
if issues:
    log_error(f"Data quality issues: {issues}")
```

---

## Deployment Best Practices

### 1. Model Versioning

**Always version your models:**
```python
# Training
result = ar.run(
    source='data_v2',
    target='default',
    ...
)

# Save with version
result.save('models/risk_model_v2.pkl')

# Deploy with version tag
ar.deploy(
    provider='azure_ml',
    name='risk-model-v2',  # Include version
    version='2.0',
    description='Updated with Q4 2024 data'
)
```

---

### 2. A/B Testing

**Deploy new model alongside old:**
```python
# Deploy v2 alongside v1
ar.deploy(
    provider='azure_ml',
    name='risk-model-v2',
    traffic_percent=10  # 10% traffic to v2, 90% to v1
)

# Monitor comparative performance
monitor_ab_test('v1', 'v2', duration_days=7)

# If v2 performs better, increase traffic
ar.update_deployment(
    name='risk-model-v2',
    traffic_percent=50  # Ramp up gradually
)
```

---

### 3. Rollback Plan

**Always have rollback capability:**
```python
# Before deploying new model
ar.backup_deployment('risk-model-v1')

# Deploy new model
ar.deploy(provider='azure_ml', name='risk-model-v2')

# If issues arise
ar.rollback_deployment('risk-model-v1')
```

---

## Regulatory Compliance

### 1. Model Documentation

**Document everything:**
```markdown
# Model Documentation: Credit Scorecard v2.0

## Business Objective
Predict default probability for personal loan applications

## Data
- Training period: Jan 2023 - Dec 2023
- Sample size: 50,000 loans
- Target: 90-day default (5.2% positive class)

## Features (15 selected)
1. debt_to_income (IV=0.45)
2. credit_score (IV=0.38)
3. ...

## Model
- Algorithm: Logistic Regression
- AUC: 0.82 (train), 0.80 (test)
- KS: 0.45
- Gini: 0.60

## Validation
- 5-fold cross-validation
- Out-of-time validation (Q1 2024)

## Monitoring
- PSI monitored weekly
- Performance reviewed monthly
```

---

### 2. Reason Codes (Adverse Action)

**Always provide reason codes:**
```python
# Score with reasons
scores = ar.score(
    'applications.csv',
    output='with_reasons'
)

for applicant in scores:
    if applicant['score'] < 650:  # Declined
        print(f"Application declined for {applicant['id']}")
        print("Primary reasons:")
        for reason in applicant['reasons'][:4]:  # Top 4 reasons
            print(f"  - {reason['reason']}")
```

**Required by law in many jurisdictions!**

---

### 3. Fairness Testing

**Check for bias:**
```python
from autoriskml.fairness import check_fairness

# Test for protected class disparities
fairness_report = check_fairness(
    data=test_data,
    predictions=predictions,
    protected_attributes=['gender', 'race', 'age_group']
)

if fairness_report.has_disparate_impact():
    log_warning("Potential fairness issue detected")
    review_model()
```

---

## Performance Optimization

### 1. Batch Scoring

**Score in batches for large datasets:**
```python
# For millions of records
scores = ar.score(
    'large_file.csv',
    chunk_size=10000,  # Process 10k at a time
    output='scores'  # Minimal output for speed
)
```

---

### 2. Caching

**Cache stable computations:**
```python
# Cache WOE tables and bins
result = ar.run(source='data', target='default')
result.save('artifacts/model_v1.pkl')

# Reuse in production
ar.load('artifacts/model_v1.pkl')
scores = ar.score('new_data.csv')  # Reuses cached WOE tables
```

---

### 3. Distributed Scoring (Future)

```python
# Coming soon: Dask/Ray support
ar.score(
    'huge_dataset.csv',
    distributed='dask',
    n_workers=8
)
```

---

## Common Pitfalls

### ❌ DON'T: Use future information

```python
# BAD: This leaks future information!
df['days_since_last_payment']  # Don't include if predicting at application time
```

### ✅ DO: Use only information available at prediction time

```python
# GOOD: Only application-time features
features = ['age', 'income', 'credit_score', 'loan_amount']
```

---

### ❌ DON'T: Ignore class imbalance

```python
# BAD: Default model on 1% positive class
result = ar.run(source='data', target='default')  # Terrible performance!
```

### ✅ DO: Balance classes

```python
# GOOD: Use class weights
config = {'models': {'logistic': {'class_weight': 'balanced'}}}
result = ar.run(source='data', target='default', config=config)
```

---

### ❌ DON'T: Overfit with too many features

```python
# BAD: 100 features for 1000 samples
config = {'features': {'max_features': 100}}
```

### ✅ DO: Use appropriate feature count

```python
# GOOD: Rule of thumb: N features <= N samples / 50
config = {'features': {'max_features': 20}}  # For 1000 samples
```

---

## Checklist

**Before Training:**
- [ ] Data profiled and quality checked
- [ ] Features engineered with domain knowledge
- [ ] Class imbalance addressed
- [ ] Train/validation/test split created (stratified)
- [ ] No data leakage verified

**During Training:**
- [ ] Appropriate binning method selected (monotonic for credit)
- [ ] IV thresholds set (min_iv >= 0.02)
- [ ] Model parameters tuned
- [ ] Cross-validation performed
- [ ] Out-of-time validation passed

**Before Deployment:**
- [ ] Model documented completely
- [ ] Reason codes generated and tested
- [ ] Fairness testing completed
- [ ] Performance benchmarks established
- [ ] Monitoring plan defined
- [ ] Rollback plan prepared
- [ ] Regulatory approval obtained (if required)

**In Production:**
- [ ] PSI monitored weekly
- [ ] AUC tracked monthly
- [ ] Data quality checked daily
- [ ] Alerts configured
- [ ] Model versioned
- [ ] Backup maintained

---

## Support

Need help? We're here!

- **Documentation**: https://github.com/idrissbado/AutoRiskML/tree/main/docs
- **Issues**: https://github.com/idrissbado/AutoRiskML/issues
- **Discussions**: https://github.com/idrissbado/AutoRiskML/discussions
- **Email**: idrissbadoolivier@gmail.com

---

**Remember: A good model is not just accurate—it's explainable, fair, and reliable in production!**
