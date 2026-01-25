This file contains notes for the deliverables asked in the requirements.md for each section wise.

## 2. Data Collection and Ingestion

### Deliverables

• Python scripts for ingestion
• Logs showing ingestion success/failure
• Folder/bucket structure for raw data

#### 2.1 Clickstream Data Ingestion
**Purpose:** Captures near-real-time user behavior (product views, search terms, session activity). This is critical for session-based recommendations and detecting immediate user intent.

**Mechanism (Stream):**
- **Source:** Apache Kafka topic `reco_clickstream`. 
- **Process:** The `clickstream_consumer.py` script acts as a persistent listener. It polls messages from Kafka, aggregates them into micro-batches (either by size or timeout), and persists them to the raw zone.
- **Storage:** Data is stored in JSON format, partitioned by date (`YYYY-MM-DD/`) to allow efficient daily batch processing.

**Deliverables:**
- `data_ingestion/clickstream_consumer.py` - Kafka consumer for clickstream events.
- `raw_zone/clickstream/YYYY-MM-DD/` - Folder structure for raw clickstream data.
- `*.json.manifest` - Manifest files containing MD5 checksum, event count, and timestamp for each JSON batch.
- Logs (via Python logging) showing ingestion success/failure.

#### 2.2 Product External Metadata Ingestion
**Purpose:** Enriches the internal product catalog with dynamic external data such as popularity scores, sentiment from reviews, or market trends. This improves the "discovery" aspect of the recommendation engine.

**Mechanism (Batch Pull):**
- **Source:** External REST API (simulated by `product_external_metadata_server.py`).
- **Process:** The `pull_product_external_metadata.py` script performs an HTTP GET request to the API. It includes a retry mechanism (3 attempts with exponential-like backoff) to handle transient network issues.
- **Storage:** The structured JSON response is converted into a Flat CSV (`raw_zone/external_metadata.csv`) for easy joining during the transformation phase.

**Deliverables:**
- `data_ingestion/pull_product_external_metadata.py` - Script to fetch product metadata from external API.
- `raw_zone/external_metadata.csv` - Target file for external metadata.
- `raw_zone/external_metadata.csv.manifest` - Manifest file for metadata integrity.
- Logs (via Python logging) showing success/failure of API calls.

#### 2.3 Transactional Data Ingestion
**Purpose:** The backbone of collaborative filtering. It records confirmed purchases (User-Item-Rating/Action). This is the "gold standard" signal for what users actually like.

**Mechanism (Batch File-Based):**
- **Source:** CSV files uploaded to a "Landing Zone" (simulated by upstream ERP/E-commerce backend dumps).
- **Process:** `ingest_transactions.py` monitors `raw_zone/landing/`. It reads new CSVs, validates the presence of timestamps, and partitions the records by the date of the transaction.
- **Storage:** Files are moved to `raw_zone/transactions/YYYY-MM-DD/`. Original source files are moved to `raw_zone/landing/archive/` to ensure idempotency and prevent double-processing.

**Deliverables:**
- `data_ingestion/ingest_transactions.py` - Script to process transactions from landing zone.
- `raw_zone/transactions/YYYY-MM-DD/` - Daily partitioned transaction files.
- `raw_zone/archive/transactions/` - Consolidated archive for processed transaction files.
- `ingested_*.csv.manifest` - Manifest files for each partitioned transaction file.

#### 2.4 Master Data Ingestion (Catalog & Customers)
**Purpose:** Provides the structural "Global IDs" and static attributes for the entire system. Without this, transactions cannot be linked to specific customers or product categories.

**Mechanism (Static Sync):**
- **Source:** Master Database exports (simulated by source files in `data_simulator/`).
- **Process:** `ingest_master_data.py` performs a direct sync/copy. It also ensures the entire `raw_zone` directory hierarchy, including the consolidated `archive/` structure, is initialized.
- **Storage:** Stored as base master files (`recomart_product_catalog.csv` and `recomart_raw_customers.csv`).

**Deliverables:**
- `data_ingestion/ingest_master_data.py` - Script to sync master files from source.
- `raw_zone/recomart_product_catalog.csv` - Product catalog master file.
- `raw_zone/recomart_raw_customers.csv` - Customer master file.
- `raw_zone/archive/transactions/` & `raw_zone/archive/landing/` - Initialized archive directories.

## 3. Raw Data Storage

### Implementation Overview
The raw data storage layer (Data Lake) is implemented using a **structured local filesystem** strategy. This mimics a cloud object storage (like S3) hierarchy, optimized for batch and stream processing tools.

#### 3.1 Storage Structure & Partitioning
- **Base Directory:** `raw_zone/`
- **Partitioning Strategy:** Data is partitioned primarily by **Source Type** and then by **Ingestion Date (`YYYY-MM-DD`)**.
- **Granularity:**
  - **Clickstream:** Partitioned by Date and further by Hour (embedded in filename) to handle high-frequency streaming events.
  - **Transactions:** Partitioned by Transaction Date to facilitate daily batch retraining.

#### 3.2 Key Directories
| Path | Data Type | Format | Partitioning |
| :--- | :--- | :--- | :--- |
| `raw_zone/clickstream/` | User behavioral events | JSON | `YYYY-MM-DD/` |
| `raw_zone/transactions/` | Purchase history | CSV | `YYYY-MM-DD/` |
| `raw_zone/archive/` | Raw source files | CSV/JSON | `SourcePackage/` |
| `raw_zone/` (Base) | Master Data/Metadata | CSV | Static/Latest |

#### 3.3 Data Integrity & Metadata
Every ingested file is accompanied by a **Manifest (`.manifest`)** file. This ensures:
- **Atomicity:** Presence of manifest indicates successful file write.
- **Auditability:** Stores row/event counts and ingestion timestamps.
- **Reliability:** MD5 checksums enable verification for downstream transformation tasks.

### Deliverables

- **Storage Structure Documentation:** Detailed above.
- **Initialization Script:** `data_ingestion/ingest_master_data.py` (Handles directory provisioning).
- **Consolidated Archive:** Unified storage for raw source files under `raw_zone/archive/`.

## 4.Data Profiling and Validation

• Apply validation checks to ensure data quality and completeness:
    ◦ Missing values, duplicate entries, schema mismatch
    ◦ Range and format checks (e.g., rating scale 1–5)
• Generate a data quality report summarizing key metrics and issues.
### Deliverables:
• Python code for automated validation (using pandas, great_expectations, or pydeequ)
• Data Quality Report (PDF)