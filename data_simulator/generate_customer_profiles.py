from pathlib import Path
import pandas as pd
import random
import ast

# ----------------------------------
# CONFIG
# ----------------------------------
BASE_DIR = Path(__file__).parent
INPUT_CUSTOMERS_FILE = BASE_DIR / "recomart_master_customers.csv"
INPUT_PRODUCTS_FILE = BASE_DIR / "recomart_product_catalog.csv"
OUTPUT_PROFILES_FILE = BASE_DIR / "recomart_user_profiles.csv"


# ----------------------------------
# PROFILE BUILDER
# ----------------------------------
def build_user_profile(user_row, all_categories):
    # Randomly select preferred categories
    preferred_categories = random.sample(all_categories, k=random.randint(1, min(3, len(all_categories))))
    
    # Generate probabilities
    base_purchase_prob = round(random.uniform(0.02, 0.12), 2)
    trend_affinity = round(random.uniform(0.8, 1.5), 2)
    
    return {
        "customer_id": user_row["customer_id"],
        "age": user_row["age"],
        "gender": user_row["gender"],
        "preferred_categories": "|".join(preferred_categories), # Storing as pipe-separated string
        "base_purchase_prob": base_purchase_prob,
        "trend_affinity": trend_affinity
    }

def main():
    print("Loading data...")
    try:
        customers_df = pd.read_csv(INPUT_CUSTOMERS_FILE)
        products_df = pd.read_csv(INPUT_PRODUCTS_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Extract unique categories
    if "category" not in products_df.columns:
        print("Error: 'category' column missing in products file.")
        return
        
    all_categories = products_df["category"].dropna().unique().tolist()
    print(f"Found categories: {all_categories}")

    # Build profiles
    print("Generating profiles...")
    profiles = []
    for _, row in customers_df.iterrows():
        profiles.append(build_user_profile(row, all_categories))
        
    profiles_df = pd.DataFrame(profiles)
    
    # Save to CSV
    profiles_df.to_csv(OUTPUT_PROFILES_FILE, index=False)
    print(f"User profiles saved to {OUTPUT_PROFILES_FILE}")

if __name__ == "__main__":
    main()
