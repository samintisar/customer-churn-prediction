# 🚀 Quick Deployment Reference

**Your Logistic Regression model is ready to deploy!**

Model Performance: **ROC-AUC 0.846** | **75% Top-Decile Precision** | **80% Recall**

---

## ⚡ Quick Start (Choose One)

### 1. Interactive Dashboard 🎨
```bash
# Install Streamlit (if needed)
pip install streamlit

# Launch dashboard
streamlit run app/dashboard.py
```
**Access:** http://localhost:8501

**Best for:** Demos, internal tools, stakeholder presentations

---

### 2. REST API 🔌
```bash
# Install Flask (if needed)
pip install flask

# Start API server
python app/api.py
```
**Access:** http://localhost:5000

**Best for:** Production integrations, automated workflows

**Test:**
```bash
curl http://localhost:5000/health
```

---

### 3. Python Script 🐍
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/logistic_regression.pkl')
feature_engineer = joblib.load('models/feature_engineer.pkl')

# Prepare data
customer_df = pd.DataFrame([{
    'customerID': '1234',
    'tenure': 1,
    'Contract': 'Month-to-month',
    'MonthlyCharges': 70.35,
    # ... other features
}])

# Transform and predict
X = customer_df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
X_transformed = feature_engineer.transform(X)
churn_prob = model.predict_proba(X_transformed)[0, 1]

print(f"Churn Risk: {churn_prob:.2%}")
```

**Best for:** Custom integrations, notebooks, automation

---

### 4. Batch Predictions 📊
```bash
# Process entire customer base
python scripts/batch_predict.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --output results/predictions.csv

# Get top 100 highest-risk customers
python scripts/batch_predict.py \
  --input data/customers.csv \
  --output results/top_100_risk.csv \
  --top-n 100
```

**Best for:** Regular scoring, monthly reports, bulk processing

---

## 📋 Quick Setup Checklist

Run this command to check everything:
```bash
python scripts/deploy_model.py --all
```

Manual checks:
- [ ] Model file exists: `models/logistic_regression.pkl` ✅
- [ ] Feature engineer exists: `models/feature_engineer.pkl` ✅
- [ ] Active model set: `models/churn_predictor.pkl` ✅
- [ ] Flask installed: `pip install flask`
- [ ] Streamlit installed: `pip install streamlit`

---

## 🌐 Cloud Deployment

### Option A: Streamlit Cloud (Easiest)
```bash
git push origin main
```
1. Go to https://share.streamlit.io
2. Connect your GitHub repo
3. Deploy `app/dashboard.py`
4. Get public URL

### Option B: Heroku (API)
```bash
# Create Procfile
echo "web: gunicorn app.api:app" > Procfile

# Deploy
heroku create your-churn-api
git push heroku main
```

### Option C: Docker (Any Platform)
```bash
docker build -t churn-predictor .
docker run -p 5000:5000 churn-predictor
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| `DEPLOYMENT_GUIDE.md` | Complete deployment instructions |
| `examples/README.md` | Code examples and usage |
| `README.md` | Project overview |
| `reports/business_report.md` | Business insights |

---

## 🎯 API Endpoints (Flask)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch predictions |

### Example Request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "tenure": 1,
    "Contract": "Month-to-month",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

---

## 🔧 Common Commands

```bash
# Switch to Logistic Regression model
python scripts/deploy_model.py --switch-to-lr

# Test API setup
python scripts/deploy_model.py --test-api

# Test dashboard setup
python scripts/deploy_model.py --test-dashboard

# Run simple example
python examples/simple_prediction.py

# Install all dependencies
pip install -r requirements.txt
```

---

## 🎓 Model Files

```
models/
├── logistic_regression.pkl      ⭐ Production model (ROC-AUC: 0.846)
├── random_forest.pkl             Alternative model
├── churn_predictor.pkl           Active model (used by dashboard)
└── feature_engineer.pkl          Feature transformation pipeline
```

---

## 💡 Tips

1. **Start with Streamlit Dashboard** - Easiest way to demo the model
2. **Use Flask API for Production** - Best for real integrations
3. **Batch Script for Reports** - Process entire customer base regularly
4. **Monitor Performance** - Track predictions vs actual churn

---

## ⚠️ Troubleshooting

### "Model not found"
```bash
# Check if models exist
ls models/

# Retrain if needed
jupyter notebook notebooks/02_model_experiments.ipynb
```

### "Module not found"
```bash
# Install missing packages
pip install -r requirements.txt
```

### "Feature mismatch"
Make sure customer data has all required features. See `examples/simple_prediction.py` for complete feature list.

---

## 📞 Need Help?

1. Check `DEPLOYMENT_GUIDE.md` for detailed instructions
2. Review examples in `examples/simple_prediction.py`
3. See API documentation in `app/api.py`

---

**Ready to deploy? Choose your method above and get started!** 🚀
