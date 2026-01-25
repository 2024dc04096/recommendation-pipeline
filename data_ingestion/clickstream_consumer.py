import argparse
import json
import os
import time
import hashlib
import logging
from datetime import datetime
from kafka import KafkaConsumer
from pathlib import Path


# ----------------------------------
# CONFIG
# ----------------------------------
KAFKA_TOPIC = "reco_clickstream"
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
RAW_ZONE = Path("raw_zone/clickstream")
BATCH_SIZE = 10  # Reduced for debug
BATCH_TIMEOUT = 10 # Reduced for debug

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_manifest(file_path: Path, event_count: int):
    """Generate a manifest file with MD5 checksum and metadata."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    manifest_path = file_path.with_suffix(file_path.suffix + '.manifest')
    manifest_content = {
        "file": file_path.name,
        "md5": md5_hash.hexdigest(),
        "event_count": event_count,
        "ingested_at": datetime.now().isoformat()
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest_content, f, indent=2)
    
    logging.info(f"Generated manifest: {manifest_path}")

def get_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",
        value_deserializer=lambda x: x, # Read raw bytes, deserialize manually
        group_id="clickstream_archiver" # Static group ID for production
    )

def save_batch(events):
    if not events:
        return
    
    # Partition by ingest time (or event time if preferred)
    # Using ingest time for simplicity of archiving
    now = datetime.now()
    date_path = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H")
    
    output_dir = RAW_ZONE / date_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"events_{hour}_{int(time.time())}.json"
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(events, f)
        
    logging.info(f"Saved {len(events)} events to {output_path}")
    generate_manifest(output_path, len(events))


def run_ingestion(timeout=None):
    print(f"Starting ingestion from {KAFKA_TOPIC}...")
    if timeout:
        print(f"Running for {timeout} seconds.")
        
    consumer = get_consumer()
    
    batch = []
    last_flush_time = time.time()
    start_time = time.time()
    
    try:
        # Use poll() instead of iteration to allow checking timeout more frequently if needed,
        # but simple iteration with timeout check inside loop is fine if volume is low-moderate.
        # For stricter control, we can use consumer.poll().
        
        while True:
            # Check global timeout
            if timeout and (time.time() - start_time > timeout):
                print("Timeout reached. Stopping ingestion.")
                break
                
            # Poll for messages (1 second timeout to allow loop check)
            msg_dict = consumer.poll(timeout_ms=1000) 
            
            if not msg_dict:
                if time.time() - last_flush_time > 5: # Print heartbeat every 5s
                     print("Waiting for messages...", end="\r")
                
                # No messages, check flush for time-based batching
                if batch and (time.time() - last_flush_time >= BATCH_TIMEOUT):
                    save_batch(batch)
                    batch = []
                    last_flush_time = time.time()
                continue
            
            for partition, messages in msg_dict.items():
                for message in messages:
                    try:
                         # Manual deserialization
                         event = json.loads(message.value.decode("utf-8"))
                         print(f"Received: {event.get('event_type')} {event.get('item_id')}...", end="\r")
                         batch.append(event)
                    except json.JSONDecodeError:
                         print(f"\nSkipping invalid JSON message at offset {message.offset}: {message.value}")
                         continue
                    except Exception as e:
                         print(f"\nError processing message: {e}")
                         continue
            
            # Check flush conditions
            time_since_flush = time.time() - last_flush_time
            if len(batch) >= BATCH_SIZE or time_since_flush >= BATCH_TIMEOUT:
                save_batch(batch)
                batch = []
                last_flush_time = time.time()
                
    except KeyboardInterrupt:
        print("Stopping ingestion...")
    finally:
        # Flush remaining
        save_batch(batch)
        consumer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=None, help="Seconds to run before exiting")
    args = parser.parse_args()
    
    run_ingestion(timeout=args.timeout)
