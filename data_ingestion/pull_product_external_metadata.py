import requests
import pandas as pd
import time
from pathlib import Path
import logging

# Config
API_URL = "http://localhost:8000/product_stats"
OUTPUT_PATH = Path("raw_zone/external_metadata.csv")
RETRIES = 3
DELAY = 5 # seconds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_external_metadata():
    """Fetches metadata from mock API and saves to raw_zone."""
    attempt = 0
    while attempt < RETRIES:
        try:
            logging.info(f"Fetching metadata from {API_URL} (Attempt {attempt + 1})...")
            response = requests.get(API_URL, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUTPUT_PATH, index=False)
            
            logging.info(f"Successfully saved metadata to {OUTPUT_PATH}")
            return True
            
        except requests.exceptions.RequestException as e:
            attempt += 1
            logging.error(f"Error fetching metadata: {e}")
            if attempt < RETRIES:
                logging.info(f"Retrying in {DELAY} seconds...")
                time.sleep(DELAY)
            else:
                logging.error("Max retries reached. Ingestion failed.")
                return False

if __name__ == "__main__":
    fetch_external_metadata()
