import json
import csv
import os

# Input and output paths
INPUT_FILE = "data/av_data_new.json"   # your saved ES response file
OUTPUT_DIR = "output_av"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Navigate to the aggregation buckets
stocks = data["aggregations"]["by_stock"]["buckets"]

print(f"Found {len(stocks)} tickers in the JSON file.")

# Process each ticker
for stock_bucket in stocks:
    ticker = stock_bucket["key"]
    daily_buckets = stock_bucket["daily_stats"]["buckets"]

    output_file = os.path.join(OUTPUT_DIR, f"{ticker}_data_av.csv")

    # Define CSV columns
    fieldnames = ["date", "avg_vol", "avg_price"]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for day_bucket in daily_buckets:
            writer.writerow({
                "date": day_bucket["key_as_string"],
                "avg_vol": day_bucket.get("avg_vol", {}).get("value"),
                "avg_price": day_bucket.get("avg_price", {}).get("value")
            })

    print(f"✅ {ticker}: wrote {len(daily_buckets)} rows to {output_file}")

print("🎉 Done! All CSV files saved in:", OUTPUT_DIR)
