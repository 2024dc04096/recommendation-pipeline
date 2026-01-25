
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os

# Definition of default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Determine the project root
# If running in Docker as per our instructions, the project is mounted at /opt/airflow/project_root
# Otherwise, we assume standard relative structure
if os.path.exists("/opt/airflow/project_root"):
    PROJECT_ROOT = "/opt/airflow/project_root"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define virtual environment path inside the project root
VENV_PATH = f"{PROJECT_ROOT}/airflow_env"
PYTHON_BIN = f"{VENV_PATH}/bin/python"

with DAG(
    'recommendation_system_pipeline',
    default_args=default_args,
    description='End-to-end recommendation system pipeline',
    schedule=timedelta(minutes=5),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['recommendation', 'mlops'],
) as dag:

    # Task 1: Setup Virtual Environment & Install Dependencies
    setup_env = BashOperator(
        task_id='setup_environment',
        bash_command=f'''
        if [ ! -d "{VENV_PATH}" ]; then
            python -m venv {VENV_PATH}
        fi
        {PYTHON_BIN} -m pip install --upgrade pip
        {PYTHON_BIN} -m pip install mlflow scikit-learn pandas kafka-python requests fpdf2 seaborn matplotlib numpy joblib
        ''',
        cwd=PROJECT_ROOT
    )

    # Task 2a: Ingest Master Data (Sync from Simulator)
    ingest_master = BashOperator(
        task_id='ingest_master_data',
        bash_command=f'{PYTHON_BIN} data_ingestion/ingest_master_data.py',
        cwd=PROJECT_ROOT
    )

    # Task 2b: Ingest Product External Metadata
    ingest_api = BashOperator(
        task_id='pull_product_external_metadata',
        bash_command=f'{PYTHON_BIN} data_ingestion/pull_product_external_metadata.py',
        cwd=PROJECT_ROOT
    )

    # Task 2c: Ingest Transactions (Landing Zone)
    ingest_txns = BashOperator(
        task_id='ingest_transactions',
        bash_command=f'{PYTHON_BIN} data_ingestion/ingest_transactions.py',
        cwd=PROJECT_ROOT
    )

    # Task 3: Data Validation
    validate_data = BashOperator(
        task_id='validate_data_quality',
        bash_command=f'{PYTHON_BIN} data_validation.py',
        cwd=PROJECT_ROOT
    )

    # Task 4: Data Transformation (Includes Cleaning & EDA)
    transform_data = BashOperator(
        task_id='transform_data',
        bash_command=f'{PYTHON_BIN} transform.py',
        cwd=PROJECT_ROOT
    )

    # Task 5: Feature Store Creation
    build_features = BashOperator(
        task_id='build_feature_store',
        bash_command=f'{PYTHON_BIN} feature_store.py',
        cwd=PROJECT_ROOT
    )

    # Task 6: Model Training
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'{PYTHON_BIN} train.py',
        cwd=PROJECT_ROOT
    )

    # Task Dependencies
    setup_env >> [ingest_master, ingest_api, ingest_txns] >> validate_data >> transform_data >> build_features >> train_model


