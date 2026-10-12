# 📦 E-Commerce Delivery Delay Risk Predictor & REST API

An end-to-end, production-grade machine learning microservice that calculates real-time shipping delay risks for e-commerce checkout orders. Built on the **Olist Brazilian E-Commerce Dataset** (~100k orders), this project processes raw relational data, engineers spatial and volumetric features, trains an imbalanced gradient boosting model, and serves predictions via a **FastAPI** backend and an interactive **Streamlit** dashboard.

---

## 💡 Business Problem & Context

In e-commerce logistics, delayed deliveries lead to increased customer support tickets, refund requests, and churn. However, predicting delays at the exact moment of order placement is challenging due to extreme class imbalance (only **~8.1%** of orders experience delays).

This project addresses the problem by:
1. Transforming raw relational tables into spatial and physical shipping risk indicators.
2. Training a lightweight **Gradient Boosting** model tuned for severe class imbalance.
3. Serving predictions via a microservice architecture so logistics teams can flag high-risk orders for priority carrier handling before dispatch.

---

## 🏗️ Architecture & Project Structure

The project strictly follows modular engineering principles, completely separating data transformation, model execution, API routes, and UI layers:

```text
delivery-delay-predictor/
├── data/
│   ├── raw/                 # Raw Olist relational CSV files
│   └── processed/           # Transformed feature matrix
├── models/
│   └── delay_model.joblib   # Serialized model & feature metadata artifact
├── src/
│   ├── __init__.py          # Package marker
│   ├── preprocessing.py     # Automated table joins, Haversine distance, & clean feature extraction
│   └── train.py             # Model training, class-weight balancing, evaluation, & serialization
├── app/
│   ├── main.py              # FastAPI microservice with Pydantic V2 validation & lifespan handlers
│   └── ui.py                # Interactive Streamlit frontend dashboard (with cloud fallback support)
├── Dockerfile               # Multi-stage Docker container specification
├── requirements.txt         # Production dependency lockfile
└── README.md                # Project documentation
```

---

## 📊 Machine Learning Pipeline & Metrics

### Feature Engineering Highlights
* **Spatial Distance (`distance_km`):** Calculated using the **Haversine formula** between customer and seller geolocation coordinates.
* **Route Scope (`is_same_state`):** Binary indicator checking if shipping crosses state borders.
* **Cargo Footprint (`product_volume_cm3`):** Volumetric size computed from mean package length, width, and height.
* **Temporal Triggers:** Extracted day-of-week (`purchase_dow`) and hour (`purchase_hour`) to capture weekend and late-night fulfillment bottlenecks.

### Model Performance & Evaluation
* **Model:** `HistGradientBoostingClassifier` with `class_weight='balanced'`.
* **Strategy:** Prioritizes catching late deliveries (high recall on the minority class) while working under severe class imbalance.

| Metric | Score | Context / Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.6849** | Strong discriminative ability relative to random baseline (0.50). |
| **PR-AUC** | **0.1807** | **$2.2\times$ baseline random chance** (0.081) on heavily imbalanced test data. |
| **Delayed Recall** | **0.61** | Captures **61% of all actual late shipments** at checkout time. |

---

## ⚡ REST API Endpoints (FastAPI)

The microservice is powered by **FastAPI** utilizing modern **Pydantic V2** schemas and `asynccontextmanager` lifespan event handlers.

### Key Endpoints

* **`GET /`**: Health-check endpoint verifying model artifact initialization status.
* **`POST /predict`**: Accepts order payloads and returns real-time delay probabilities and risk tier labels.

#### Sample Request Payload (`POST /predict`)
```json
{
  "total_freight_value": 18.50,
  "total_item_price": 120.00,
  "num_items": 1,
  "product_weight_g": 800.0,
  "product_volume_cm3": 3000.0,
  "distance_km": 450.2,
  "is_same_state": 0,
  "estimated_days": 14.0,
  "purchase_dow": 4,
  "purchase_hour": 14
}
```

#### Sample API Response (`200 OK`)
```json
{
  "delay_probability": 0.5137,
  "is_high_risk": true,
  "risk_level": "HIGH"
}
```

---

## 🖥️ Interactive Web Dashboard

The **Streamlit UI** (`app/ui.py`) offers a user-friendly interface for logistics managers to simulate order inputs, adjust cargo attributes via interactive sliders, and receive instant color-coded risk alerts (`LOW`, `MEDIUM`, `HIGH`).

* **Hybrid Execution:** The dashboard communicates directly with the FastAPI REST server via HTTP requests locally, and automatically falls back to native model execution when deployed on cloud hosting environments like Streamlit Community Cloud.

---

## 🚀 Getting Started Locally

### 1. Prerequisites & Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/dineshkrishna-264/delivery-delay-predictor.git
cd delivery-delay-predictor

# Create and activate virtual environment
python -m venv venv

# On Mac/Linux:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Processing & Model Training
Execute the modular scripts inside `src/`:

```bash
# Process raw CSV data into feature matrix
python src/preprocessing.py

# Train model, evaluate metrics, and save artifact to models/
python src/train.py
```

### 3. Launch the FastAPI Microservice
Start the Uvicorn server:

```bash
python -m uvicorn app.main:app --reload
```
View the auto-generated Swagger UI documentation by navigating to `http://127.0.0.1:8000/docs`.

### 4. Launch the Streamlit Frontend
In a second terminal window (with virtual environment activated):

```bash
streamlit run app/ui.py
```
Open `http://localhost:8501` in your browser to interact with the dashboard.

---

## 🐳 Containerization (Docker)

To run the application inside an isolated Docker container:

```bash
# Build the Docker image
docker build -t delivery-delay-api .

# Run the container mapping port 8000
docker run -p 8000:8000 delivery-delay-api
```

---

## 🛠️ Tech Stack

* **Language & Core:** Python 3.10+, Pandas, NumPy, Scikit-Learn, Joblib
* **API Microservice:** FastAPI, Uvicorn, Pydantic V2
* **Frontend UI:** Streamlit, Requests
* **DevOps & Packaging:** Docker, Git
