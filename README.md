
# 🏥 SmartED AI System

An end-to-end AI system for **Emergency Department (ED) forecasting and risk analysis**, integrating:

* Data pipeline (automated ingestion + feature engineering)
* Time series model (Multi-target LSTM)
* Risk engine (rule-based decision logic)
* LLM explanation layer (Azure OpenAI)
* API services and cloud deployment (Azure)

---

# 🧠 System Overview

```text
Data Pipeline → ADLS → Model API → Risk API → LLM → Streamlit Dashboard
```

This system not only predicts ED pressure, but also **translates predictions into operational risks and actionable insights**.

---

# 📂 Project Structure

```text
AzureDemo_version4/
│
├── api_main.py                  # Risk API (LLM + risk engine)
├── config.py                   # Global configuration
│
├── model_artifacts/            # (ignored in GitHub)
│
├── services/                  # Core logic
│   ├── adls_reader.py
│   ├── inference.py
│   ├── model_client.py
│   ├── risk_engine.py
│   ├── llm_explainer.py
│   ├── forecast_service.py    # Orchestrator
│   └── missing_checker.py
│
├── data_pipeline/             # Data processing
│   ├── main.py
│   ├── sources/
│   ├── transform/
│   ├── storage/
│   └── config.py
│
├── Dockerfile.model_api
├── Dockerfile.risk_api
├── Dockerfile.streamlit
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Data Pipeline

## Overview

The pipeline automatically collects and processes data daily:

### Data Sources

* HSE Daily ED report (trolley / waiting)
* HSE Weekly attendance (PDF)
* Weather data (Open-Meteo API)

---

## Data Processing Flow

### 1. Daily Master Row (Data Integration)

* Merge:

  * Daily report
  * Weekly attendance
  * Weather data
* Align by date
* Add calendar features:

  * day_of_week
  * is_weekend
  * is_holiday
  * month

👉 Output: **one row per day**

---

### 2. Latest Features (Feature Engineering)

Built from historical data:

* Lag features (t-1, t-7, etc.)
* Trend features
* Rolling statistics (mean / std)
* Seasonal encoding (week sin/cos)
* Missing handling (ffill / bfill)

👉 Output: **model-ready feature table**

---

### 3. History Storage

* Append new daily record
* Deduplicate by date
* Store in ADLS

---

### 4. Data Layers

| Layer     | Purpose       |
| --------- | ------------- |
| raw       | raw ingestion |
| processed | cleaned data  |
| serving   | model input   |

---

# 🤖 Model (Multi-target LSTM)

## Objective

Predict 3 targets simultaneously:

* `uhl_ed`
* `uhl_wait_24h`
* `uhl_wait_75plus`

---

## Architecture

```text
LSTM (2 layers)
    ↓
Fully Connected
    ↓
3 outputs
```

---

## Key Features

* Multi-target learning
* Sequence input (e.g., past 14 days)
* Recursive multi-step forecasting
* Early stopping + gradient clipping

---

## Metrics

```text
uhl_ed           | MAE ≈ 2.05
uhl_wait_24h     | MAE ≈ 6.35
uhl_wait_75plus  | MAE ≈ 3.17
```

---

# 🔌 API Design

## 1. Model API

**Endpoint:**

```text
POST /predict
```

**Function:**

* Load latest features from ADLS
* Run LSTM inference
* Output 7-day forecast

---

## 2. Risk Control API

**Endpoint:**

```text
GET /forecast-risk-control
```

**Function:**

* Call Model API
* Apply rule-based thresholds
* Generate LLM explanation

---

## Output Includes

* Daily risk (Low / Medium / High)
* Overall risk (72h / 7d)
* Peak day and driver
* Recommended actions
* LLM explanation

---

## Example Output

```json
{
  "overall_risk_72h": "Medium",
  "peak_day": "2026-04-13",
  "peak_driver": "24-hour waiting pressure",
  "llm_explanation": {
    "executive_summary": "...",
    "recommended_actions": [...]
  }
}
```

---

# 🧠 LLM Explanation Layer

Uses Azure OpenAI to convert predictions into:

* Executive summary
* Risk drivers
* Action recommendations
* Monitoring suggestions

👉 Purpose:

```text
Make model output interpretable and decision-oriented
```

---

# ☁️ Deployment (Azure)

## Architecture

* Azure Container Apps (API hosting)
* Azure Container Registry (images)
* Azure Data Lake (data storage)
* Azure OpenAI (LLM)

---

## Services

* Model API
* Risk API
* Streamlit UI

---

## Deployment Workflow

```text
Build Docker → Push to ACR → Deploy Container App → Set env vars
```

---

# 📊 Frontend (Streamlit)

* Forecast visualization
* Risk dashboard
* LLM explanation display
* Daily risk timeline

---

# 🚀 Key Highlights

```text
✔ End-to-end AI system
✔ Real-world healthcare use case
✔ Time series forecasting
✔ Rule-based + LLM hybrid decision system
✔ Production-style cloud deployment
```

---

# 📌 Future Work

* Alert system
* Hospital integration
* More data to build an specific healthcare agent 

---

