import pandas as pd
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import logging

# Config
LANDING_ZONE = Path("raw_zone/landing")
TARGET_DIR = Path("raw_zone/transactions")
# Consolidated archive location (top-level with source subdirectory)
ARCHIVE_DIR = Path("raw_zone/archive/transactions")
CATALOG_FILE = Path("raw_zone/recomart_product_catalog.csv")
CUSTOMERS_FILE = Path("raw_zone/recomart_raw_customers.csv")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_manifest(file_path: Path, row_count: int):
    """Generate a manifest file with MD5 checksum and metadata."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    manifest_path = file_path.with_suffix(file_path.suffix + '.manifest')
    manifest_content = {
        "file": file_path.name,
        "md5": md5_hash.hexdigest(),
        "row_count": row_count,
        "ingested_at": datetime.now().isoformat()
    }
    
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest_content, f, indent=2)
    
    logging.info(f"Generated manifest: {manifest_path}")

def ingest_from_landing():
    """Moves transactions from landing zone into partitioned folders and archives them."""
    # Ensure directories exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all CSVs in landing zone
    txn_files = list(LANDING_ZONE.glob("*.csv"))
    
    if not txn_files:
        logging.info("No new files found in landing zone.")
        return

    for txn_file in txn_files:
        logging.info(f"Processing {txn_file}...")
        try:
            df = pd.read_csv(txn_file)
            if 'txn_date' not in df.columns:
                logging.warning(f"No txn_date in {txn_file}. Using today's date.")
                date_str = datetime.now().strftime("%Y-%m-%d")
                # Save entire file to a single partition using today's date
                date_folder = TARGET_DIR / date_str
                date_folder.mkdir(parents=True, exist_ok=True)
                target_file = date_folder / f"ingested_{txn_file.name}"
                df.to_csv(target_file, index=False)
                logging.info(f"Saved file (no txn_date) to {target_file}")
                generate_manifest(target_file, len(df))
            else:
                # Convert to datetime to extract date
                df['date_parsed'] = pd.to_datetime(df['txn_date'])
                # Group by date and save separately
                for date, group in df.groupby(df['date_parsed'].dt.date):
                    date_folder = TARGET_DIR / str(date)
                    date_folder.mkdir(parents=True, exist_ok=True)
                    
                    target_file = date_folder / f"ingested_{txn_file.name}"
                    # Don't overwrite if not necessary, but for ingestion we usually replace or append
                    group.drop(columns=['date_parsed']).to_csv(target_file, index=False)
                    logging.info(f"Saved partition for {date} to {target_file}")
                    generate_manifest(target_file, len(group))

            
            # Move processed file to archive
            shutil.move(str(txn_file), str(ARCHIVE_DIR / txn_file.name))
            logging.info(f"Archived {txn_file} to {ARCHIVE_DIR}")
            
        except Exception as e:
            logging.error(f"Error processing {txn_file}: {e}")


def organize_catalog():
    """Placeholder for catalog organization if needed."""
    pass

if __name__ == "__main__":
    ingest_from_landing()
