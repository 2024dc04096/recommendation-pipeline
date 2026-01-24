import json
import os
import random
import time
import uuid
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from kafka import KafkaProducer

# ----------------------------------
# CONFIG
# ----------------------------------
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
TOPIC = "reco_clickstream"

# Robust Path Resolution
BASE_DIR = Path(__file__).parent.parent
TRANSACTIONS_LANDING = BASE_DIR / "raw_zone" / "landing"
TRANSACTIONS_LANDING.mkdir(parents=True, exist_ok=True)

EVENT_STRENGTH = {
    "view": 1,
    "add_to_cart": 3,
    "purchase": 5
}

# Transaction recording
recorded_transactions = []

DEVICE_TYPES = ["mobile", "desktop", "tablet"]

SEASONS = {
    12: "Festive",
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer"
}

SEASONAL_CATEGORY_BOOST = {
    "Winter": {"Electronics": 1.3, "Fashion": 1.2, "Home": 1.0},
    "Summer": {"Fashion": 1.3, "Home": 1.1, "Electronics": 1.0},
    "Festive": {"Electronics": 1.5, "Fashion": 1.4, "Home": 1.2}
}

# ----------------------------------
# KAFKA PRODUCER
# ----------------------------------
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)



# ----------------------------------
# TRENDING PRODUCTS
# ----------------------------------
def mark_trending_products(products, trend_ratio=0.2):
    trending = set(
        p["product_id"] for p in random.sample(
            products, int(len(products) * trend_ratio)
        )
    )
    return trending

# ----------------------------------
# PRODUCT SAMPLING (SEASON + TREND)
# ----------------------------------
def sample_product(products, user_profile, season, trending_products):
    weights = []

    for p in products:
        weight = 1.0

        # User preference
        if p["category"] in user_profile["preferred_categories"]:
            weight *= 1.5

        # Seasonality
        weight *= SEASONAL_CATEGORY_BOOST.get(season, {}).get(p["category"], 1.0)

        # Trending boost
        if p["product_id"] in trending_products:
            weight *= user_profile["trend_affinity"]

        weights.append(weight)

    return random.choices(products, weights=weights, k=1)[0]

# ----------------------------------
# EVENT GENERATOR
# ----------------------------------
def generate_event(user_profile, product, event_type, session_id, timestamp, season):
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_profile["user_id"],
        "item_id": product["product_id"],
        "category": product["category"],
        "price": product["price"],
        "event_type": event_type,
        "event_strength": EVENT_STRENGTH[event_type],
        "session_id": session_id,
        "timestamp": timestamp.isoformat(),
        "season": season,
        "day_of_week": timestamp.strftime("%A"),
        "hour_of_day": timestamp.hour,
        "is_trending": product["product_id"] in trending_products,
        "device": random.choice(DEVICE_TYPES),
        "dwell_time": int(np.random.normal(22, 7)) if event_type == "view" else None
    }

# ----------------------------------
# SESSION SIMULATOR
# ----------------------------------
def simulate_session(user_profile, products, start_time, trending_products):
    session_id = str(uuid.uuid4())
    season = SEASONS.get(start_time.month, "Normal")
    current_time = start_time
    events = []

    for _ in range(random.randint(3, 7)):
        product = sample_product(products, user_profile, season, trending_products)

        events.append(generate_event(
            user_profile, product, "view", session_id, current_time, season
        ))
        current_time += timedelta(seconds=random.randint(5, 20))

        if random.random() < 0.3:
            events.append(generate_event(
                user_profile, product, "add_to_cart", session_id, current_time, season
            ))
            current_time += timedelta(seconds=random.randint(5, 15))

            seasonal_purchase_prob = user_profile["base_purchase_prob"] * \
                SEASONAL_CATEGORY_BOOST.get(season, {}).get(product["category"], 1.0)

            if random.random() < seasonal_purchase_prob:
                events.append(generate_event(
                    user_profile, product, "purchase", session_id, current_time, season
                ))
                break

    return events

# ----------------------------------
# MAIN LOOP
# ----------------------------------
def generate_reco_data(user_profiles, products, days=30, sessions_per_day=2, skip_kafka=False):
    global trending_products
    trending_products = mark_trending_products(products)

    start_date = datetime.now() - timedelta(days=days)

    for day in range(days):
        for profile in user_profiles:
            for _ in range(sessions_per_day):
                session_start = start_date + timedelta(
                    days=day,
                    hours=random.randint(8, 23)
                )

                events = simulate_session(
                    profile, products, session_start, trending_products
                )

                for event in events:
                    if not skip_kafka and producer:
                        producer.send(TOPIC, event)
                        print("Sent to Kafka:", event["event_type"], event["item_id"])
                    
                    # Record transaction if event is 'purchase'
                    if event["event_type"] == "purchase":
                        recorded_transactions.append({
                            "txn_id": f"TXN-{event['session_id'][:8].upper()}",
                            "txn_date": pd.to_datetime(event["timestamp"]).strftime("%Y-%m-%d"),
                            "customer_id": event["user_id"],
                            "product_id": event["item_id"],
                            "quantity": 1
                        })
                    
                    if not skip_kafka:
                        time.sleep(0.01) # Small delay for live feel

    # Save recorded transactions
    if recorded_transactions:
        df = pd.DataFrame(recorded_transactions)
        prefix = "live_transactions" if not skip_kafka else "historical_transactions"
        output_file = TRANSACTIONS_LANDING / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully recorded {len(recorded_transactions)} transactions to {output_file}")

# ----------------------------------
# RUN
# ----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecoMart Data Simulator")
    parser.add_argument("--days", type=int, default=30, help="Days of data to simulate")
    parser.add_argument("--skip-kafka", action="store_true", help="Only generate transaction CSV, skip Kafka")
    args = parser.parse_args()

    print(f"Starting simulation for {args.days} days (Skip Kafka: {args.skip_kafka})...")
    
    try:
        # Kafka setup (conditional)
        producer = None
        if not args.skip_kafka:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BROKER,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    request_timeout_ms=5000
                )
            except Exception as e:
                print(f"Warning: Could not connect to Kafka: {e}. Defaulting to CSV only.")
                args.skip_kafka = True

        # Load products
        products_df = pd.read_csv(BASE_DIR / "recomart_product_catalog.csv")
        if "price" not in products_df.columns and "base_price" in products_df.columns:
            products_df = products_df.rename(columns={"base_price": "price"})
        
        if "price" not in products_df.columns:
            # Assign random prices if missing
            print("Assigning random prices to products...")
            products_df["price"] = [random.randint(500, 30000) for _ in range(len(products_df))]
        
        products = products_df.to_dict("records")
        
        # Load profiles
        profiles_df = pd.read_csv(BASE_DIR / "recomart_user_profiles.csv")
        # Parse categories back to list
        profiles_df["preferred_categories"] = profiles_df["preferred_categories"].apply(lambda x: str(x).split("|"))
        
        # Rename customer_id to user_id for compatibility
        if "customer_id" in profiles_df.columns:
            profiles_df = profiles_df.rename(columns={"customer_id": "user_id"})
            
        user_profiles = profiles_df.to_dict("records")
        
        print(f"Loaded {len(user_profiles)} profiles and {len(products)} products.")

        generate_reco_data(user_profiles, products, days=args.days, skip_kafka=args.skip_kafka)
        
    except KeyboardInterrupt:
        print("\nStopping generator...")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure raw data CSVs and generated profiles exist.")
    finally:
        print("Flushing messages...")
        if producer:
            producer.flush()
            producer.close()
        print("Done.")

