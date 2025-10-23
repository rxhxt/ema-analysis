import json
import csv
import os

# Input ES response file
INPUT_FILE = "data/reddit_data_new.json"
OUTPUT_DIR = "output_reddit"

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Elasticsearch response
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract top-level aggregation buckets (one per ticker)
tickers = data["aggregations"]["by_ticker"]["buckets"]

print(f"Found {len(tickers)} tickers in the response.")

for ticker_bucket in tickers:
    ticker = ticker_bucket["key"]
    daily_buckets = ticker_bucket["daily_stats"]["buckets"]

    # Build CSV file path for this ticker
    output_file = os.path.join(OUTPUT_DIR, f"{ticker}_data_reddit.csv")

    # Define CSV columns
    fieldnames = [
        "date",
        "daily_volume",
        "avg_fsai_Sentiment_score",
        "avg_fsai_Sentiment_score_text"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for day_bucket in daily_buckets:
            row = {
                "date": day_bucket["key_as_string"],
                "daily_volume": day_bucket.get("daily_volume", {}).get("value", day_bucket["doc_count"]),
                "avg_fsai_Sentiment_score": day_bucket.get("avg_fsai_Sentiment_score", {}).get("value"),
                "avg_fsai_Sentiment_score_text": day_bucket.get("avg_fsai_Sentiment_score_text", {}).get("value")
            }
            writer.writerow(row)

    print(f"✅ Wrote {len(daily_buckets)} rows → {output_file}")

print("🎉 Done! All ticker CSVs are saved in the 'output/' folder.")
