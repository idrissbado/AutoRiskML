# AutoRiskML Documentation

Welcome to AutoRiskML - The First Fully Automated Risk & Trading Intelligence Engine!

## 📚 Table of Contents

1. [Quick Start Guide](quickstart.md) - Get started in 5 minutes ✅
2. Installation Guide (See Quick Start) ✅
3. [API Reference](api_reference.md) - Full API documentation ✅
4. [Architecture](architecture.md) - System design and components ✅
5. Tutorials (Coming Soon)
6. [Examples](examples.md) - Complete code examples ✅
7. Azure Deployment (Coming Soon)
8. [Best Practices](best_practices.md) - Production recommendations ✅
9. FAQ (Coming Soon)
10. Contributing (Coming Soon)

## 🚀 What is AutoRiskML?

AutoRiskML is a Python package that automates the entire risk modeling pipeline:

```python
from autoriskml import AutoRisk

ar = AutoRisk(project="credit_scoring")
ar.register_source("train", csv="loans.csv")
result = ar.run(
    source="train",
    target="default_flag",
    explain=True,
    deploy={"provider": "azure_ml"}
)
```

**One command does everything:**
- ✅ Data profiling & recommendations
- ✅ Automated cleaning
- ✅ Binning & WOE/IV computation
- ✅ Model training & selection
- ✅ Scorecard generation
- ✅ PSI monitoring & drift detection
- ✅ SHAP explainability
- ✅ Azure deployment

## 🎯 Key Features

### 1. **Automated Data Processing**
- Smart data profiling with recommendations
- Automatic missing value handling
- Outlier detection and treatment
- Type inference and coercion

### 2. **Risk-Specific Modeling**
- **WOE/IV Computation** - Weight of Evidence & Information Value
- **Monotonic Binning** - Optimal risk-based binning
- **Scorecard Generation** - Convert models to credit scores
- **PSI/CSI Monitoring** - Population Stability Index tracking

### 3. **Machine Learning**
- Multiple model support (Logistic, XGBoost, LightGBM)
- Hyperparameter tuning
- Walk-forward validation for time-series
- Model calibration

### 4. **Explainability**
- SHAP global and local explanations
- Reason code generation
- Feature importance analysis
- Model interpretability reports

### 5. **Monitoring & Alerts**
- Drift detection
- PSI threshold alerts
- Performance degradation detection
- Automatic retrain triggers

### 6. **Production Deployment**
- Azure ML integration
- AKS deployment
- FastAPI endpoint generation
- Containerization support

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  AutoRisk API                        │
│          (High-level user interface)                 │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│            Pipeline Orchestrator                     │
│  • Stage execution                                   │
│  • Artifact management                               │
│  • Provenance tracking                               │
└──┬───────┬────────┬─────────┬────────┬────────┬────┘
   │       │        │         │        │        │
   ▼       ▼        ▼         ▼        ▼        ▼
┌───────┬───────┬───────┬───────┬────────┬──────────┐
│Connec │Profile│Cleaning│Binning│ Models │  Scoring │
│-tors  │       │        │WOE/IV │Training│Scorecard │
└───────┴───────┴────────┴───────┴────────┴──────────┘
                     │
        ┌────────────┼───────────────────┐
        ▼            ▼                   ▼
   ┌────────┐  ┌──────────┐      ┌────────────┐
   │Metrics │  │ Explain  │      │ Monitoring │
   │PSI/CSI │  │SHAP/LIME │      │Drift/Alert │
   └────────┘  └──────────┘      └────────────┘
                     │
        ┌────────────┴──────────────────┐
        ▼                               ▼
   ┌─────────┐                    ┌──────────┐
   │ Export  │                    │Deployment│
   │ONNX/    │                    │Azure ML/ │
   │Joblib   │                    │AKS/API   │
   └─────────┘                    └──────────┘
```

## 📖 Quick Links

- **Installation:** `pip install autoriskml`
- **PyPI:** https://pypi.org/project/autoriskml/
- **GitHub:** https://github.com/idrissbado/AutoRiskML
- **Issues:** https://github.com/idrissbado/AutoRiskML/issues
- **Discussions:** https://github.com/idrissbado/AutoRiskML/discussions

## 🤝 Community

- **Questions?** Open a [Discussion](https://github.com/idrissbado/AutoRiskML/discussions)
- **Bug?** Open an [Issue](https://github.com/idrissbado/AutoRiskML/issues)
- **Feature Request?** Open an [Issue](https://github.com/idrissbado/AutoRiskML/issues)
- **Want to Contribute?** See [CONTRIBUTING.md](../CONTRIBUTING.md)

## 📧 Support

- **Author:** Idriss Bado
- **Email:** idrissbadoolivier@gmail.com
- **GitHub:** [@idrissbado](https://github.com/idrissbado)

## 📄 License

MIT License - see [LICENSE](../LICENSE) file

---

**Ready to get started? Check out the [Quick Start Guide](quickstart.md)!**
