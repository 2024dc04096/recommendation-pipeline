import pandas as pd
import numpy as np
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Config
RAW_PATH = Path("raw_zone")
TXN_PARTITIONS = RAW_PATH / "transactions"
CLICKSTREAM_PATH = RAW_PATH / "clickstream"
EXTERNAL_METADATA = RAW_PATH / "external_metadata.csv"
CATALOG_FILE = RAW_PATH / "recomart_product_catalog.csv"
CUSTOMERS_FILE = RAW_PATH / "recomart_raw_customers.csv"

OUT_DIR = Path("data/processed")
EDA_PATH = OUT_DIR / "eda"

# Ensure output directories exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
EDA_PATH.mkdir(parents=True, exist_ok=True)

def preprocess_catalog(df):
    """Clean, Encode, and Normalize product catalog."""
    print("Preprocessing Catalog (Encoding & Scaling)...")
    df = df.copy()
    
    # Fill missing
    cols_to_fill_zero = ["discount_percent", "monthly_sales_volume", "return_rate_percent"]
    for c in cols_to_fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Label Encoding (Categorical)
    cat_cols = ["category", "brand", "is_perishable", "super_category"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            
    # MinMax Scaling (Numerical)
    num_cols = ["base_price", "monthly_sales_volume", "avg_rating", "return_rate_percent"]
    scaler = MinMaxScaler()
    for col in num_cols:
        if col in df.columns:
            df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
            
    return df

def preprocess_customers(df):
    """Clean and process customer data."""
    print("Preprocessing Customers...")
    df = df.copy()
    
    df = df.drop_duplicates(subset=["customer_id"])
    
    # Encode Gender
    if "gender" in df.columns:
        le = LabelEncoder()
        df["gender_encoded"] = le.fit_transform(df["gender"].astype(str))
        
    # Scale Age
    if "age" in df.columns:
        scaler = MinMaxScaler()
        df["age_scaled"] = scaler.fit_transform(df[["age"]])
        
    return df

def perform_eda(transactions, catalog):
    """Generate EDA plots and stats."""
    print("Performing Exploratory Data Analysis...")
    
    # 1. Sparsity
    n_users = transactions["customer_id"].nunique()
    n_items = transactions["product_id"].nunique()
    n_interactions = len(transactions)
    matrix_size = n_users * n_items
    sparsity = 1 - (n_interactions / matrix_size) if matrix_size > 0 else 0
    
    print(f"  - Users: {n_users}")
    print(f"  - Items: {n_items}")
    print(f"  - Interactions: {n_interactions}")
    print(f"  - Sparsity: {sparsity:.4%}")
    
    # 2. Item Popularity (Top 20)
    plt.figure(figsize=(12, 6))
    
    if not catalog.empty and 'popularity_index' in catalog.columns:
        # Use dynamic popularity index
        top_items_df = catalog.sort_values('popularity_index', ascending=False).head(20)
        labels = [f"{row['product_name'][:20]}..." for _, row in top_items_df.iterrows()]
        values = top_items_df['popularity_index']
        metric_name = "Popularity Index"
    else:
        # Fallback to pure transaction counts
        top_items = transactions["product_id"].value_counts().head(20)
        if not catalog.empty and "product_name" in catalog.columns:
            names = catalog.set_index("product_id")["product_name"]
            labels = [names.get(pid, pid) for pid in top_items.index]
        else:
            labels = top_items.index
        values = top_items.values
        metric_name = "Number of Transactions"
        
    sns.barplot(x=values, y=labels, palette="viridis")
    plt.title(f"Top 20 Popular Items ({metric_name})")
    plt.xlabel(metric_name)
    plt.tight_layout()
    plt.savefig(EDA_PATH / "item_popularity.png")
    plt.close()
    
    # 3. User Interaction Dist (Log scale)
    plt.figure(figsize=(10, 5))
    user_counts = transactions["customer_id"].value_counts()
    sns.histplot(user_counts, bins=3, kde=True)
    plt.title("User Interaction Distribution (Log Scale)")
    plt.xlabel("Interactions per User")
    plt.savefig(EDA_PATH / "user_activity_dist.png")
    plt.close()
    
    print(f"  - EDA plots saved to {EDA_PATH}")

def transform():
    print("Loading partitioned transactions...")
    txn_files = list(TXN_PARTITIONS.glob("**/ingested_*.csv"))
    if not txn_files:
        print("Warning: No partitioned transactions found. Checking for legacy...")
        legacy = list(RAW_PATH.glob("*transaction*.csv"))
        if legacy:
            monthly_txns = pd.read_csv(legacy[0])
        else:
            print("Error: No transaction data found.")
            return
    else:
        monthly_txns = pd.concat([pd.read_csv(f) for f in txn_files], ignore_index=True)

    # Static data
    customers_raw = pd.read_csv(CUSTOMERS_FILE)
    products_raw = pd.read_csv(CATALOG_FILE)
    
    # Preprocess Static Data (Consolidated from data_preprocessing.py)
    customers = preprocess_customers(customers_raw)
    products = preprocess_catalog(products_raw)
    
    # Save cleaned static data
    customers.to_csv(OUT_DIR / "clean_customers.csv", index=False)
    products.to_csv(OUT_DIR / "clean_product_catalog.csv", index=False)
    print(f"Cleaned static data saved to {OUT_DIR}")

    # External metadata (Mock API output)
    if EXTERNAL_METADATA.exists():
        metadata = pd.read_csv(EXTERNAL_METADATA)
        print(f"Loaded {len(metadata)} external metadata records.")
    else:
        metadata = pd.DataFrame()

    # Drop records with if record does not contains either customer_id or product_id
    monthly_txns = monthly_txns[monthly_txns['customer_id'].notna() & monthly_txns['product_id'].notna()]
    
    # # Clean numeric values
    monthly_txns['quantity'] = pd.to_numeric(monthly_txns['quantity'], errors='coerce')

    # Merge with static data
    monthly_txns = pd.merge(monthly_txns, customers, on='customer_id', how='left')
    monthly_txns = pd.merge(monthly_txns, products, on='product_id', how='inner') # Inner join to filter ghost products
    
    print(f"Transactions after filtering invalid products: {len(monthly_txns)}")
    
    # Merge with external metadata
    if not metadata.empty:
        monthly_txns = pd.merge(monthly_txns, metadata, on='product_id', how='left')
    
    # Calculate price features
    # Use base_price from cleaned products
    monthly_txns['product_price'] = monthly_txns['base_price'] - (monthly_txns['base_price'] * (monthly_txns['discount_percent'].fillna(0) / 100))
    monthly_txns['total_price'] = monthly_txns['product_price'] * monthly_txns['quantity']

    # Discretize age column (numeric age is handled in preprocess_customers)
    if 'age' in monthly_txns.columns:
        monthly_txns['age_group'] = pd.cut(monthly_txns['age'], bins=[0, 18, 35, 50, 65, 80], labels=['0-18', '19-35', '36-50', '51-65', '66-80'])

    # Process Clickstream and Create Unified Interactions
    clickstream_df = process_clickstream()
    
    # Create unified interactions (Transaction interactions + Clickstream interactions)
    # Let's say purchase in transaction = 5 pts per quantity
    txn_interactions = monthly_txns[['customer_id', 'product_id', 'quantity']].copy()
    txn_interactions['interaction_score'] = txn_interactions['quantity'] * 5 
    
    # Merge
    if not clickstream_df.empty:
        # Filter clickstream for valid products only
        valid_product_ids = products['product_id'].unique()
        clickstream_df = clickstream_df[clickstream_df['product_id'].isin(valid_product_ids)]
        
        unified = pd.concat([txn_interactions, clickstream_df], ignore_index=True)
    else:
        unified = txn_interactions
        
    # Aggregate scores (sum) if multiple entries for same user-item
    unified = unified.groupby(['customer_id', 'product_id'], as_index=False)['interaction_score'].sum()
    
    # --- DYNAMIC POPULARITY CALCULATION ---
    print("Calculating dynamic popularity scores...")
    prod_popularity = unified.groupby('product_id')['interaction_score'].sum().reset_index()
    prod_popularity.columns = ['product_id', 'raw_pop_score']
    
    # Apply Log scaling to handle long-tail distributions
    prod_popularity['log_pop'] = np.log1p(prod_popularity['raw_pop_score'])
    
    # Scale to 10 - 1000 range
    min_log = prod_popularity['log_pop'].min()
    max_log = prod_popularity['log_pop'].max()
    
    if max_log > min_log:
        prod_popularity['popularity_index'] = 10 + (prod_popularity['log_pop'] - min_log) / (max_log - min_log) * 990
    else:
        prod_popularity['popularity_index'] = 500 # Default if all same
        
    prod_popularity['popularity_index'] = prod_popularity['popularity_index'].round(0).astype(int)
    
    # Merge new popularity back into dataframes
    # Remove old mock popularity if it exists
    if 'popularity_index' in monthly_txns.columns:
        monthly_txns = monthly_txns.drop(columns=['popularity_index'])
    if 'popularity_index' in products.columns:
        products = products.drop(columns=['popularity_index'])
        
    monthly_txns = pd.merge(monthly_txns, prod_popularity[['product_id', 'popularity_index']], on='product_id', how='left')
    products = pd.merge(products, prod_popularity[['product_id', 'popularity_index']], on='product_id', how='left')
    
    # Fill products with 0 interactions with a minimum popularity
    monthly_txns['popularity_index'] = monthly_txns['popularity_index'].fillna(10)
    products['popularity_index'] = products['popularity_index'].fillna(10)
    # --------------------------------------

    # Save outputs
    monthly_txns.to_csv(OUT_DIR/"transactions_enriched.csv", index=False)
    products.to_csv(OUT_DIR / "clean_product_catalog.csv", index=False)
    unified.to_csv(OUT_DIR/"unified_interactions.csv", index=False)
    
    print("Enriched transaction data saved →", OUT_DIR/"transactions_enriched.csv")
    print("Cleaned product catalog saved →", OUT_DIR/"clean_product_catalog.csv")
    print("Unified interactions saved →", OUT_DIR/"unified_interactions.csv")

    # Final Step: Perform EDA on enriched transactions
    perform_eda(monthly_txns, products)


def process_clickstream():
    print("Processing clickstream data...")
    json_files = glob.glob(str(CLICKSTREAM_PATH / "**/*.json"), recursive=True)
    
    if not json_files:
        print("No clickstream files found.")
        return pd.DataFrame(columns=['customer_id', 'product_id', 'interaction_score'])
        
    all_events = []
    for f in json_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                all_events.extend(data)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_events:
        return pd.DataFrame(columns=['customer_id', 'product_id', 'interaction_score'])
        
    df = pd.DataFrame(all_events)
    
    # Map events to scores
    # event_strength is already in the data from generator, but let's enforce local logic or use it.
    # Generator: view=1, add_to_cart=3, purchase=5
    # The generator provides 'event_strength', we can just use it.
    
    # Aggregate: Sum strength per user-item
    # Rename user_id to customer_id to match transactions
    df = df.rename(columns={'user_id': 'customer_id', 'item_id': 'product_id'})
    
    interactions = df.groupby(['customer_id', 'product_id'], as_index=False)['event_strength'].sum()
    interactions = interactions.rename(columns={'event_strength': 'interaction_score'})
    
    return interactions

if __name__ == "__main__":
    transform()