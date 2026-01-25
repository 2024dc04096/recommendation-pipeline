import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import random
import csv
from pathlib import Path

# Config
PORT = 8000
BASE_DIR = Path(__file__).parent
PRODUCTS_CSV = BASE_DIR / "recomart_product_catalog.csv"

class MockAPIHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        if self.path == '/product_stats':
            self._set_headers()
            stats = self._generate_product_stats()
            self.wfile.write(json.dumps(stats).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def _generate_product_stats(self):
        """Generates realistic stats aligned with product catalog."""
        stats = []
        print(f"API: Looking for catalog at {PRODUCTS_CSV.absolute()}")
        if not PRODUCTS_CSV.exists():
            return [{"error": "Product catalog not found"}]

        with open(PRODUCTS_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['product_id']
                # Seed with pid for deterministic results per item
                random.seed(pid)
                
                sentiment = round(random.uniform(0.1, 1.0), 2)
                
                stats.append({
                    "product_id": pid,
                    "sentiment_score": sentiment,
                    "last_updated": "2026-01-18"
                })
        return stats

def run(server_class=HTTPServer, handler_class=MockAPIHandler, port=PORT):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting Mock API on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Stopping Mock API...")

if __name__ == "__main__":
    run()
