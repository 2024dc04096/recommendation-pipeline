# RecoMart Recommendation Pipeline

An end-to-end MLOps pipeline for a product recommendation system, featuring hybrid batch-stream ingestion, automated validation, feature store management, and live performance monitoring.

## 🚀 Quick Start Guide

Follow these steps to run the entire pipeline from scratch.

### 1. Prerequisites & Environment
- **Docker & Docker Compose** (for Kafka/Zookeeper)
- **Python 3.10+**
```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Infrastructure & Seed Data
Launch the Kafka broker and generate initial user profiles:

```bash
# Start Kafka & Zookeeper
docker-compose up -d

# Generate User Profiles (Run once to seed the simulator)
cd data_simulator
python generate_customer_profiles.py
cd ..
```

### 3. Start the Always-On Layer
Launch the data simulators and the persistent stream consumer. **Open separate terminals for each:**

- **Terminal A: Mock API (Product Stats)**
  ```bash
  python data_simulator/product_external_metadata_server.py
  ```
- **Terminal B: Clickstream Simulator (Kafka Events)**
  ```bash
  python data_simulator/clickstream_data_generator.py --days 1
  ```
- **Terminal C: Stream Consumer**
  ```bash
  python data_ingestion/clickstream_consumer.py
  ```

### 4. Run the Data Pipeline
You can trigger the pipeline either via Airflow or by manually executing the scripts.

#### A. Manual Execution (Recommended for Testing)
Run these in order to sync master data, ingest additions, validate, and train:

```bash
# 1. Sync Master Data & Create Folders (CRITICAL FIRST STEP)
# This pulls master Catalog/Customers from simulator to raw_zone
python data_ingestion/ingest_master_data.py

# 2. Ingest API Metadata & Process Transaction Landing Zone
python data_ingestion/pull_product_external_metadata.py
python data_ingestion/ingest_transactions.py

# 3. Validate Data Quality
# Generates "recomart_data_quality_report.pdf"
python data_validation.py

# 4. Transform Data & Build Feature Store
python transform.py
python feature_store.py

# 5. Train Models & Monitor
python train.py
python monitor.py
```

#### B. Airflow Orchestration
If you have Airflow set up:
1. Enable the `recommendation_system_pipeline` DAG.
2. The DAG handles the entire sequence from `ingest_master_data` to `train_model`.

---

### 5. View Results & Analytics

- **Streamlit Dashboard**:
  ```bash
  python -m streamlit run app.py
  ```
  *Features: Personalized Recommendations, Training Metrics, and Production Health Monitoring.*

- **MLflow Tracking**:
  ```bash
  mlflow ui
  ```

---

## 🛠️ Components & Monitoring
- **Data Quality**: Automated PDF reports showing schema health and referential integrity.
- **Drift Detection**: `monitor.py` tracks changes in data distribution between training and inference.
- **Inference Logging**: Every recommendation made in the UI is logged to `model_store/inference_logs.db` for latency and accuracy audit.

## 🔍 Troubleshooting
- **IndexError in Dashboard**: This usually happens if the Feature Store is out of sync with the Metadata Catalog (e.g., after deduplication). 
  - **Fix**: Re-run the full pipeline: `Sync -> Ingest -> Validate -> Transform -> Store -> Train`.
  - **Hot Reload**: If Streamlit doesn't pick up code changes, stop it (`Ctrl+C`) and start again: `python -m streamlit run app.py`.

## 🔍 Key Verification Commands
- **Check Master Data**: `ls raw_zone/recomart_product_catalog.csv`
- **Check Transactions**: `ls -R raw_zone/transactions/`
- **Generate 30-Day Bulk History**: 
  `python data_simulator/clickstream_data_generator.py --skip-kafka --days 30`
  (Then run the ingestion pipeline to process the landing files)
