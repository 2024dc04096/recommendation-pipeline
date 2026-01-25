import random
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Setup
DB_PATH = Path("model_store/feature_store.db")
MODEL_STORE = Path("model_store")
MODEL_STORE.mkdir(exist_ok=True)

mlflow.set_experiment("Recommendation_System_Experiment")

def load_data():
    conn = sqlite3.connect(DB_PATH)
    interactions = pd.read_sql("SELECT * FROM interactions", conn)
    try:
        # Use the specific metadata table created by Member 2
        products = pd.read_sql("SELECT * FROM product_metadata", conn)
    except:
        products = pd.DataFrame()
        print("Warning: product_metadata table not found. Using fallback.")
    conn.close()
    return interactions, products

def calculate_precision_at_k(actual, predicted, k=10):
    """Calculates Precision@K for a list of recommendations."""
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if not act_set: return 0
    return len(act_set.intersection(pred_set)) / float(k)

def train_collaborative_filtering(interactions):
    print("Training Collaborative Filtering (TruncatedSVD)...")
    
    # Pivot for User-Item Matrix
    user_item_matrix = interactions.pivot(index='customer_id', columns='product_id', values='interaction_score').fillna(0)
    user_map = {idx: uid for idx, uid in enumerate(user_item_matrix.index)}
    item_map = {idx: iid for idx, iid in enumerate(user_item_matrix.columns)}
    
    X = user_item_matrix.values
    
    with mlflow.start_run(run_name="Collaborative_Filtering"):
        n_components = 20
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(X)
        item_factors = svd.components_.T # Product embeddings
        
        reconstructed = np.dot(user_factors, item_factors.T)
        
        # Calculate standard regression metrics
        mask = X > 0
        rmse = np.sqrt(mean_squared_error(X[mask], reconstructed[mask]))
        mlflow.log_metric("rmse", rmse)
        print(f"SVD RMSE: {rmse:.4f}")

        # Calculate Ranking Metrics (Precision@10 and Recall@10)
        # We sample 100 users for speed in this demo
        sample_users = random.sample(range(X.shape[0]), min(100, X.shape[0]))
        precisions = []
        recalls = []
        
        for u_idx in sample_users:
            actual_items = np.where(X[u_idx, :] > 5)[0] # Threshold for 'actual' interest
            if len(actual_items) == 0: continue
            
            # Predict scores for all items
            pred_scores = reconstructed[u_idx, :]
            top_k_indices = np.argsort(pred_scores)[::-1][:10]
            
            # Intersection
            hits = len(set(actual_items).intersection(set(top_k_indices)))
            
            p_at_k = hits / 10.0
            r_at_k = hits / len(actual_items)
            
            precisions.append(p_at_k)
            recalls.append(r_at_k)
            
        avg_p_at_10 = np.mean(precisions) if precisions else 0
        avg_r_at_10 = np.mean(recalls) if recalls else 0
        
        mlflow.log_metric("precision_at_10", avg_p_at_10)
        mlflow.log_metric("recall_at_10", avg_r_at_10)
        
        print(f"Avg Precision@10: {avg_p_at_10:.4f}")
        print(f"Avg Recall@10: {avg_r_at_10:.4f}")

        # Save artifacts
        joblib.dump(svd, MODEL_STORE / "svd_model.pkl")
        joblib.dump(user_item_matrix, MODEL_STORE / "user_item_matrix.pkl")
        joblib.dump(interactions, MODEL_STORE / "interactions.pkl")
        mlflow.sklearn.log_model(svd, "model")
        
    return user_item_matrix, reconstructed

def train_content_based(products):
    print("\nTraining Content-Based Filtering...")
    
    # Enrichment: Use Product Name + Category + Brand
    products['soup'] = products['product_name'].fillna('') + " " + \
                       products['category'].fillna('') + " " + \
                       products['brand'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['soup'])
    
    # Linear kernel is same as cosine_similarity for normalized TF-IDF
    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    with mlflow.start_run(run_name="Content_Based"):
        mlflow.log_param("n_products", len(products))
        
        # Log mean similarity (density check)
        mean_sim = np.mean(cosine_sim)
        mlflow.log_metric("mean_item_similarity", mean_sim)
        
        # Save artifacts
        joblib.dump(products, MODEL_STORE / "products.pkl")
        joblib.dump(cosine_sim, MODEL_STORE / "cosine_sim.pkl")
        print(f"CB Similarity Matrix calculated ({cosine_sim.shape})")

def main():
    import random
    interactions, products = load_data()
    
    if not interactions.empty:
        train_collaborative_filtering(interactions)
    
    if not products.empty:
        train_content_based(products)

if __name__ == "__main__":
    main()
