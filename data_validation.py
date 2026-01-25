import os
import json
import pandas as pd
from datetime import datetime
from dateutil.parser import parse as dtparse
from fpdf import FPDF

# ------------------------------------------------------------
# Optional Great Expectations (guarded)
# ------------------------------------------------------------
GE_ENABLED = False
GE_CTX = None
Validator = None
PandasExecutionEngine = None

try:
    from great_expectations.validator.validator import Validator
    from great_expectations.execution_engine.pandas_execution_engine import PandasExecutionEngine
    GE_ENABLED = True
except Exception as e:
    print(f"DEBUG: Great Expectations could not be imported: {e}")
    GE_ENABLED = False

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PATH = "raw_zone/"
FILES = {
    "product_catalog": PATH + "recomart_product_catalog.csv",
    "raw_customers": PATH + "recomart_raw_customers.csv",
    "external_metadata": PATH + "external_metadata.csv"
}

# Transaction files are now partitioned
TXN_PATH = PATH + "transactions/"

# Ensure structure exists
os.makedirs(TXN_PATH, exist_ok=True)
os.makedirs(os.path.join(PATH, "clickstream"), exist_ok=True)
os.makedirs(os.path.join(PATH, "landing"), exist_ok=True)

CLICKSTREAM_PATH = os.path.join(PATH, "clickstream")

EXPECTED_SCHEMAS = {
    "product_catalog": {
        "columns": [
            "product_id","product_name","super_category","category","brand",
            "base_price","discount_percent","monthly_sales_volume","avg_rating",
            "return_rate_percent","profit_margin_percent","is_perishable",
            "shelf_life_days","launch_date"
        ],
        "types": {
            "product_id": "str","product_name": "str","super_category": "str",
            "category": "str","brand": "str","base_price": "numeric",
            "discount_percent": "numeric","monthly_sales_volume": "numeric",
            "avg_rating": "numeric","return_rate_percent": "numeric",
            "profit_margin_percent": "numeric","is_perishable": "str",
            "shelf_life_days": "numeric","launch_date": "date",
        },
    },
    "raw_customers": {
        "columns": ["customer_id","age","gender"],
        "types": {"customer_id": "str","age": "numeric","gender": "str"},
    },
    "raw_products": {
        "columns": ["product_id","product_name","category"],
        "types": {"product_id": "str","product_name": "str","category": "str"},
    },
    "transactions": {
        "columns": ["txn_id","txn_date","customer_id","product_id","quantity"],
        "types": {
            "txn_id": "str","txn_date": "date","customer_id": "str",
            "product_id": "str","quantity": "numeric",
        },
    },
    "external_metadata": {
        "columns": ["product_id", "sentiment_score", "popularity_index", "last_updated"],
        "types": {
            "product_id": "str",
            "sentiment_score": "numeric",
            "popularity_index": "numeric",
            "last_updated": "date"
        },
    },
    "clickstream": {
        "columns": [
            "event_id", "user_id", "item_id", "category", "price", "event_type",
            "event_strength", "session_id", "timestamp", "device"
        ],
        "types": {
            "event_id": "str", "user_id": "str", "item_id": "str",
            "event_type": "str", "event_strength": "numeric",
            "timestamp": "date"
        }
    }
}

RULES = {
    "product_catalog": {
        "base_price": {"min": 0, "max": 10000},
        "discount_percent": {"min": 0, "max": 100},
        "avg_rating": {"min": 1, "max": 5},
        "return_rate_percent": {"min": 0, "max": 100},
        "profit_margin_percent": {"min": 0, "max": 100},
        "shelf_life_days": {"min": 0, "max": 3650},
        "is_perishable": {"allowed": ["Yes","No"]},
        "launch_date": {"format": "%Y-%m-%d"},
    },
    "raw_customers": {
        "age": {"min": 18, "max": 100},
        "gender": {"allowed": ["M","F","O"]},
    },
    "transactions": {
        "quantity": {"min": 1, "max": 1000},
        "txn_date": {"format": "%Y-%m-%d"},
    },
    "external_metadata": {
        "sentiment_score": {"min": 0, "max": 1},
        "popularity_index": {"min": 0, "max": 10000},
        "last_updated": {"format": "%Y-%m-%d"},
    },
    "clickstream": {
        "event_strength": {"min": 1, "max": 5},
        "event_type": {"allowed": ["view", "add_to_cart", "purchase"]},
        "device": {"allowed": ["mobile", "tablet", "desktop"]},
    }
}

# ------------------------------------------------------------
# Helpers (pandas)
# ------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def check_schema(df: pd.DataFrame, expected_cols: list) -> dict:
    actual = list(df.columns)
    missing = [c for c in expected_cols if c not in actual]
    extra = [c for c in actual if c not in expected_cols]
    return {"missing_columns": missing, "extra_columns": extra}

def is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna())
        return True
    except Exception:
        return False

def range_check(s: pd.Series, min_v=None, max_v=None) -> dict:
    s_num = pd.to_numeric(s, errors="coerce")
    issues = []
    if min_v is not None:
        bad_min = s_num.dropna()[s_num.dropna() < min_v]
        if len(bad_min) > 0:
            issues.append(f"{len(bad_min)} values < {min_v}")
    if max_v is not None:
        bad_max = s_num.dropna()[s_num.dropna() > max_v]
        if len(bad_max) > 0:
            issues.append(f"{len(bad_max)} values > {max_v}")
    return {"issues": issues}

def format_check_date(s: pd.Series, fmt: str) -> dict:
    bad = []
    for i, v in s.dropna().items():
        try:
            datetime.strptime(str(v), fmt)
        except Exception:
            bad.append(i)
    return {"bad_count": len(bad), "bad_indices": bad[:10]}

def categorical_check(s: pd.Series, allowed: list) -> dict:
    invalid = s.dropna()[~s.dropna().isin(allowed)]
    return {"invalid_count": len(invalid), "examples": invalid.head(10).tolist()}

def duplicates_check(df: pd.DataFrame, key_cols: list) -> dict:
    dup = df.duplicated(subset=key_cols, keep=False)
    return {"duplicate_rows": int(dup.sum())}

def missing_values(df: pd.DataFrame) -> dict:
    mv = df.isna().sum()
    return {"missing_by_column": mv[mv > 0].to_dict(), "total_missing": int(mv.sum())}

def referential_integrity(child: pd.DataFrame, parent: pd.DataFrame, child_key: str, parent_key: str) -> dict:
    missing_refs = child[~child[child_key].isin(parent[parent_key])]
    return {"missing_ref_count": int(len(missing_refs)), "examples": missing_refs.head(10).to_dict(orient="records")}

# ------------------------------------------------------------
# Optional GE validation (modern API, guarded)
# ------------------------------------------------------------
def ge_validate(df: pd.DataFrame, dataset_name: str) -> dict:
    if not GE_ENABLED:
        return {"ge_enabled": False, "ge_success": None, "ge_statistics": {}, "ge_results": {}}

    try:
        import great_expectations as ge
        ge_df = ge.from_pandas(df)

        if dataset_name == "product_catalog":
            ge_df.expect_column_values_to_not_be_null("product_id")
            ge_df.expect_column_values_to_be_unique("product_id")
            ge_df.expect_column_values_to_be_between("base_price", 0, 10000)
            ge_df.expect_column_values_to_be_between("discount_percent", 0, 100)
            ge_df.expect_column_values_to_be_between("avg_rating", 1, 5)
            ge_df.expect_column_values_to_be_between("return_rate_percent", 0, 100)
            ge_df.expect_column_values_to_be_between("profit_margin_percent", 0, 100)
            ge_df.expect_column_values_to_be_in_set("is_perishable", ["Yes","No"])
            ge_df.expect_column_values_to_match_strftime("launch_date", "%Y-%m-%d")

        elif dataset_name == "raw_customers":
            ge_df.expect_column_values_to_not_be_null("customer_id")
            ge_df.expect_column_values_to_be_unique("customer_id")
            ge_df.expect_column_values_to_be_between("age", 18, 100)
            ge_df.expect_column_values_to_be_in_set("gender", ["M","F","O"])

        elif dataset_name == "raw_products":
            ge_df.expect_column_values_to_not_be_null("product_id")
            ge_df.expect_column_values_to_be_unique("product_id")
            ge_df.expect_column_values_to_not_be_null("product_name")
            ge_df.expect_column_values_to_not_be_null("category")

        elif dataset_name == "transactions":
            ge_df.expect_column_values_to_not_be_null("txn_id")
            ge_df.expect_column_values_to_be_unique("txn_id")
            ge_df.expect_column_values_to_not_be_null("customer_id")
            ge_df.expect_column_values_to_not_be_null("product_id")
            ge_df.expect_column_values_to_be_between("quantity", 1, 1000)
            ge_df.expect_column_values_to_match_strftime("txn_date", "%Y-%m-%d")

        res = ge_df.validate()
        stats = res.get("statistics", {})
        return {
            "ge_enabled": True,
            "ge_success": res.get("success"),
            "ge_statistics": stats,
            "ge_results": res,
        }
    except Exception as e:
        # If GE fails for any reason, continue with pandas-only results
        print(f"DEBUG: GE Runtime Error in {dataset_name}: {e}")
        return {"ge_enabled": False, "ge_success": None, "ge_statistics": {"error": str(e)}, "ge_results": {}}

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
dfs = {}
for name, path in FILES.items():
    if os.path.exists(path):
        dfs[name] = load_csv(path)
    else:
        print(f"Warning: {path} not found.")

# Load transactions (Partitions + Legacy)
import glob
txn_dfs = []

# 1. Partitions
partition_files = glob.glob(os.path.join(TXN_PATH, "**/*.csv"), recursive=True)
if partition_files:
    print(f"Loading {len(partition_files)} transaction partitions...")
    txn_dfs.extend([pd.read_csv(f) for f in partition_files])

# 2. Legacy File


# 3. Combine
if txn_dfs:
    dfs["transactions"] = pd.concat(txn_dfs, ignore_index=True)
else:
    print("Warning: No transaction data found (partitions or legacy).")
    dfs["transactions"] = pd.DataFrame(columns=EXPECTED_SCHEMAS["transactions"]["columns"])


# 4. Load Clickstream (JSON)
clickstream_files = glob.glob(os.path.join(CLICKSTREAM_PATH, "**/*.json"), recursive=True)
clickstream_data = []

if clickstream_files:
    print(f"Loading {len(clickstream_files)} clickstream files...")
    for f in clickstream_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    clickstream_data.extend(data)
                elif isinstance(data, dict):
                     clickstream_data.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

if clickstream_data:
    dfs["clickstream"] = pd.DataFrame(clickstream_data)
else:
    print("Warning: No clickstream data found.")
    dfs["clickstream"] = pd.DataFrame(columns=EXPECTED_SCHEMAS["clickstream"]["columns"])

# ------------------------------------------------------------
# Validation pipeline
# ------------------------------------------------------------
summary = {}

for name, df in dfs.items():
    schema = check_schema(df, EXPECTED_SCHEMAS[name]["columns"])
    mv = missing_values(df)

    pk = None
    if name == "product_catalog":
        pk = ["product_id"]
    elif name == "raw_customers":
        pk = ["customer_id"]
    elif name == "raw_products":
        pk = ["product_id"]
    elif name == "transactions":
        pk = ["txn_id"]
    dup = duplicates_check(df, pk) if pk else {"duplicate_rows": None}

    rules = RULES.get(name, {})
    range_format_issues = {}

    for col, rule in rules.items():
        if col not in df.columns:
            range_format_issues[col] = {"error": "column_missing"}
            continue

        s = df[col]
        if "min" in rule or "max" in rule:
            if is_numeric_series(s):
                range_format_issues.setdefault(col, {})
                range_format_issues[col]["range"] = range_check(s, rule.get("min"), rule.get("max"))
            else:
                range_format_issues[col] = {"error": "non_numeric_values_present"}

        if "allowed" in rule:
            range_format_issues.setdefault(col, {})
            range_format_issues[col]["categorical"] = categorical_check(s, rule["allowed"])

        if "format" in rule:
            range_format_issues.setdefault(col, {})
            range_format_issues[col]["date_format"] = format_check_date(s, rule["format"])

    ge_res = ge_validate(df, name)

    summary[name] = {
        "row_count": int(len(df)),
        "schema": schema,
        "missing_values": mv,
        "duplicates": dup,
        "range_format_issues": range_format_issues,
        "ge": ge_res,
    }

# ------------------------------------------------------------
# Cross-dataset integrity checks
# ------------------------------------------------------------
integrity = {}
if "transactions" in dfs and "raw_customers" in dfs:
    integrity["txn_customer_fk"] = referential_integrity(
        dfs["transactions"], dfs["raw_customers"], "customer_id", "customer_id"
    )

if "transactions" in dfs and "product_catalog" in dfs:
    missing_in_catalog = dfs["transactions"][~dfs["transactions"]["product_id"].isin(dfs["product_catalog"]["product_id"])]
    integrity["txn_product_fk_catalog"] = {
        "missing_ref_count": int(len(missing_in_catalog)),
        "examples": missing_in_catalog.head(10).to_dict(orient="records"),
    }

if "external_metadata" in dfs and "product_catalog" in dfs:
    missing_meta = dfs["external_metadata"][~dfs["external_metadata"]["product_id"].isin(dfs["product_catalog"]["product_id"])]
    integrity["meta_product_fk"] = {
        "missing_ref_count": int(len(missing_meta)),
        "examples": missing_meta.head(10).to_dict(orient="records"),
    }

# ------------------------------------------------------------
# PDF Report (Table-based)
# ------------------------------------------------------------
class ReportPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        # fpdf 1.7.2: cell(w, h, txt, border, ln, align, fill, link)
        # ln=1 means move to next line
        self.cell(0, 10, "Recomart Data Quality Report", 0, 1, "C")
        self.set_font("helvetica", "", 10)
        self.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        self.ln(5)

    def section_title(self, title):
        self.ln(5)
        self.set_font("helvetica", "B", 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, "L", True)
        self.ln(2)
        self.set_font("helvetica", "", 10)

    def add_summary_table(self, summary_data):
        self.set_font("helvetica", "", 10)
        # Manual table construction for FPDF 1.7.2
        col_width = 38
        self.set_font("helvetica", "B", 10)
        headers = ["Dataset", "Rows", "Missing", "Dups", "GE Status"]
        for header in headers:
            self.cell(col_width, 7, header, 1, 0, 'C')
        self.ln()
        
        self.set_font("helvetica", "", 10)
        for name, stats in summary_data.items():
            self.cell(col_width, 7, str(name)[:18], 1)
            self.cell(col_width, 7, str(stats["row_count"]), 1)
            self.cell(col_width, 7, str(stats["missing_values"]["total_missing"]), 1)
            dup = stats["duplicates"]["duplicate_rows"]
            self.cell(col_width, 7, str(dup) if dup is not None else "N/A", 1)
            
            ge_status = "Enabled" if stats["ge"]["ge_enabled"] else "Disabled"
            if stats["ge"]["ge_success"] is not None:
                ge_status += " (Pass)" if stats["ge"]["ge_success"] else " (Fail)"
            self.cell(col_width, 7, ge_status, 1)
            self.ln()

    def add_dataset_details(self, name, stats):
        self.section_title(f"Dataset Details: {name}")
        
        # Schema Table
        self.set_font("helvetica", "B", 11)
        self.cell(0, 8, "Schema & Missing Values", 0, 1, "L")
        self.set_font("helvetica", "", 10)
        
        mv_cols = stats["missing_values"]["missing_by_column"]
        extra_cols = stats["schema"]["extra_columns"]
        missing_cols = stats["schema"]["missing_columns"]

        data_rows = []
        if missing_cols:
             data_rows.append(["Schema Error", f"Missing cols: {', '.join(missing_cols)}", False])
        if extra_cols:
             data_rows.append(["Schema Warning", f"Extra cols: {', '.join(extra_cols)}", False])
        
        if mv_cols:
            for col, count in mv_cols.items():
                data_rows.append([f"Column: {col}", f"Missing: {count}", False])
        else:
            data_rows.append(["Missing Values", "None detected", True])

        # Table
        w = 95
        self.set_font("helvetica", "B", 10)
        self.cell(w, 7, "Check", 1)
        self.cell(w, 7, "Result", 1)
        self.ln()
        self.set_font("helvetica", "", 10)
        
        for d in data_rows:
            self.cell(w, 7, str(d[0])[:50], 1)
            if d[2]: # Passed
                self.set_text_color(0, 128, 0)
            else: # Failed
                self.set_text_color(200, 0, 0)
            self.cell(w, 7, str(d[1])[:50], 1)
            self.set_text_color(0, 0, 0)
            self.ln()

        # Rules Table
        rf_issues = stats["range_format_issues"]
        if rf_issues:
            self.ln(4)
            self.set_font("helvetica", "B", 11)
            self.cell(0, 8, "Data Quality Rules", 0, 1, "L")
            self.set_font("helvetica", "", 10)
            
            # Table headers
            w_col = 40
            w_type = 30
            w_status = 20
            w_detail = 100
            
            self.set_font("helvetica", "B", 10)
            self.cell(w_col, 7, "Column", 1)
            self.cell(w_type, 7, "Check Type", 1)
            self.cell(w_status, 7, "Status", 1)
            self.cell(w_detail, 7, "Details", 1)
            self.ln()
            self.set_font("helvetica", "", 10)
            
            for col, issues in rf_issues.items():
                for issue_type, details in issues.items():
                    self.cell(w_col, 7, str(col)[:20], 1)
                    self.cell(w_type, 7, str(issue_type)[:15], 1)
                    
                    # Determine Pass/Fail
                    passed = False
                    detail_str = ""
                    
                    if isinstance(details, dict):
                        # range check
                        if "issues" in details:
                            if not details["issues"]:
                                passed = True
                            else:
                                detail_str = "; ".join(details["issues"])
                        # categorical
                        elif "invalid_count" in details:
                            if details["invalid_count"] == 0:
                                passed = True
                            else:
                                ex = details.get("examples", [])
                                detail_str = f"{details['invalid_count']} invalid. Ex: {ex}"
                        # date format
                        elif "bad_count" in details:
                            if details["bad_count"] == 0:
                                passed = True
                            else:
                                detail_str = f"{details['bad_count']} bad formats."
                        # generic error
                        elif "error" in details:
                            detail_str = details["error"]
                    else:
                        detail_str = str(details)
                        
                    if passed:
                        self.set_text_color(0, 128, 0)
                        self.cell(w_status, 7, "PASS", 1)
                        self.set_text_color(0, 0, 0)
                        self.cell(w_detail, 7, "-", 1)
                    else:
                        self.set_text_color(200, 0, 0)
                        self.cell(w_status, 7, "ISSUE", 1)
                        self.set_text_color(0, 0, 0)
                        self.cell(w_detail, 7, str(detail_str)[:55], 1)
                    self.ln()

    def add_integrity_section(self, integrity_data):
        self.section_title("Cross-Dataset Integrity")
        
        # Table
        w_name = 60
        w_status = 30
        w_detail = 100
        
        self.set_font("helvetica", "B", 10)
        self.cell(w_name, 7, "Relationship Check", 1)
        self.cell(w_status, 7, "Status", 1)
        self.cell(w_detail, 7, "Details", 1)
        self.ln()
        self.set_font("helvetica", "", 10)
        
        for check_name, result in integrity_data.items():
            self.cell(w_name, 7, str(check_name)[:30], 1)
            
            missing_count = result.get("missing_ref_count", 0)
            passed = missing_count == 0
            
            if passed:
                self.set_text_color(0, 128, 0)
                self.cell(w_status, 7, "PASS", 1)
                self.set_text_color(0, 0, 0)
                self.cell(w_detail, 7, "All references valid", 1)
            else:
                self.set_text_color(200, 0, 0)
                self.cell(w_status, 7, "FAIL", 1)
                self.set_text_color(0, 0, 0)
                self.cell(w_detail, 7, f"{missing_count} orphan references found", 1)
            self.ln()

pdf = ReportPDF()
pdf.add_page()

# Executive Summary
pdf.section_title("Executive Summary")
pdf.add_summary_table(summary)

# Details per dataset
for name, s in summary.items():
    pdf.add_dataset_details(name, s)

# Integrity
pdf.add_integrity_section(integrity)

OUTPUT_DIR = "data/processed/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_NAME = os.path.join(OUTPUT_DIR, "recomart_data_quality_report.pdf")
pdf.output(REPORT_NAME)

print(f"Report generated: {REPORT_NAME}")
print("\n[Summary Table]")
print(f"{'Dataset':<20} | {'Rows':<8} | {'Missing':<8} | {'Dups':<6} | {'GE Success'}")
print("-" * 70)
for name, s in summary.items():
    ge_s = s['ge']['ge_success']
    print(f"{name:<20} | {s['row_count']:<8} | {s['missing_values']['total_missing']:<8} | "
          f"{s['duplicates']['duplicate_rows'] or 'N/A':<6} | {ge_s}")

print("\n[Integrity Checks]")
for k, v in integrity.items():
    print(f"- {k}: {v.get('missing_ref_count', 'N/A')} issues")

