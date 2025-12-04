# Quick Start Guide

Get started with AutoRiskML in 5 minutes!

## Installation

```bash
# Basic installation (pure Python, zero dependencies)
pip install autoriskml

# With machine learning support
pip install autoriskml[ml]

# With explainability
pip install autoriskml[explain]

# With Azure deployment
pip install autoriskml[azure]

# Full installation (everything)
pip install autoriskml[all]
```

## Your First Risk Model (30 Seconds)

```python
from autoriskml import AutoRisk

# 1. Initialize project
ar = AutoRisk(project="my_first_model")

# 2. Register your data
ar.register_source("train", csv="path/to/your/data.csv")

# 3. Run the complete pipeline
result = ar.run(
    source="train",
    target="your_target_column",
    explain=True
)

# 4. Check results
print(f"Model AUC: {result.metrics['auc']:.3f}")
print(f"Report: {result.report_html}")
```

**That's it!** You just:
- ✅ Profiled your data
- ✅ Auto-cleaned it
- ✅ Computed WOE/IV
- ✅ Trained a model
- ✅ Generated a scorecard
- ✅ Created a report

## Complete Example: Credit Scoring

```python
from autoriskml import AutoRisk

# Create project
ar = AutoRisk(
    project="loan_scoring",
    output_dir="artifacts/loans",
    log_level="INFO"
)

# Register data sources
ar.register_source("train", csv="data/loans_2023.csv")
ar.register_source("test", csv="data/loans_2024.csv")

# Run full pipeline
result = ar.run(
    source="train",
    validation_source="test",
    target="default_flag",
    
    # Cleaning options
    clean={
        "missing_strategy": "auto",
        "outlier_method": "iqr"
    },
    
    # Binning options
    binning={
        "numeric_method": "monotonic",
        "max_bins": 6
    },
    
    # Feature selection
    features={
        "min_iv": 0.02,
        "max_features": 20
    },
    
    # Models to try
    models=["logistic", "xgboost"],
    
    # Scorecard settings
    scorecard={
        "pdo": 20,
        "base_score": 600
    },
    
    # Enable explainability
    explain=True,
    
    # Monitoring
    monitor={
        "compute_psi": True,
        "psi_threshold": 0.2
    },
    
    # Generate reports
    report={
        "formats": ["html", "pdf"]
    }
)

# Results
print(f"✅ Best Model: {result.best_model}")
print(f"✅ AUC: {result.metrics['auc']:.3f}")
print(f"✅ PSI: {result.psi['overall_psi']:.3f}")
print(f"📄 Report: {result.report_html}")
```

## Score New Customers

```python
# Score new data
scores = ar.score(
    "new_customers.csv",
    output="with_reasons"
)

# View results
for score in scores[:5]:
    print(f"Customer {score['index']}:")
    print(f"  Score: {score['score']}")
    print(f"  Risk Tier: {score['risk_tier']}")
    print(f"  Top Reason: {score['top_reason']}")
```

## Monitor Production Data

```python
# Check for drift
monitor = ar.monitor(
    source="production_data",
    baseline_source="train"
)

if monitor.alert:
    print(f"⚠️  ALERT: {monitor.message}")
    print(f"PSI: {monitor.overall_psi:.3f}")
    print(f"Drifted features: {monitor.drifted_features}")
```

## Deploy to Azure

```python
# Deploy model
result = ar.run(
    source="train",
    target="default_flag",
    deploy={
        "provider": "azure_ml",
        "workspace": "MyWorkspace",
        "resource_group": "my-rg"
    }
)

print(f"Endpoint: {result.endpoint.scoring_uri}")
print(f"Key: {result.endpoint.primary_key}")
```

## Common Use Cases

### 1. Credit Scoring
```python
ar = AutoRisk(project="credit_scoring")
ar.register_source("train", csv="loans.csv")
result = ar.run(source="train", target="default", scorecard={"pdo": 20})
```

### 2. Fraud Detection
```python
ar = AutoRisk(project="fraud_detection")
ar.register_source("train", csv="transactions.csv")
result = ar.run(source="train", target="is_fraud", models=["xgboost"])
```

### 3. Trading Risk
```python
ar = AutoRisk(project="trading_risk", mode="trading")
ar.register_source("signals", parquet="s3://bucket/signals.parquet")
result = ar.run(source="signals", target="return", backtest=True)
```

## Next Steps

1. **[API Reference](api_reference.md)** - Full API documentation
2. **[Tutorials](tutorials/)** - Step-by-step guides
3. **[Examples](examples.md)** - More complete examples
4. **[Azure Deployment](azure_deployment.md)** - Production deployment
5. **[Best Practices](best_practices.md)** - Production recommendations

## Need Help?

- 📖 [Full Documentation](README.md)
- 💬 [Discussions](https://github.com/idrissbado/AutoRiskML/discussions)
- 🐛 [Issues](https://github.com/idrissbado/AutoRiskML/issues)
- 📧 Email: idrissbadoolivier@gmail.com

---

**Ready to build production risk models? Let's go! 🚀**
