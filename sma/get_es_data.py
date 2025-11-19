import requests
import csv
import json
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
USERNAME = "sastage"
PASSWORD = "Fsaistage#1"

index = "fsai_news_text"
host = "https://vpc-ssi-stg-sog3lhbiblkri6h6mqwun7jjde.ap-south-1.es.amazonaws.com"

search_url = f"{host}/{index}/_search?scroll=2m"
scroll_url = f"{host}/_search/scroll"

auth = (USERNAME, PASSWORD)

# ==========================================
# UPDATED QUERY BODY
# ==========================================
query_body = {
    "query": {
        "bool": {
            "must": [
                {
                    "range": {
                        "Created_at": {
                            "gte": "2025-11-03",
                            "lte": "2025-11-04"
                        }
                    }
                }
            ]
        }
    },
    "_source": [
        "Created_at",
        "Body",
        "Text",
        "Ticker",
        "Sentiment_score",
        "Sentiment_score_text",
        "fsai_Sentiment_score",
        "fsai_Sentiment_score_text",
        "fsai_Sentiment_label_text",
        "fsai_Sentiment_label",
        "url",
        "fsai_sentiment",
        "fsai_Targeted_Sentiment"
    ],
    "size": 500
}

# ==========================================
# SCROLL WITH PROGRESS
# ==========================================
def scroll_all_data():
    print("🚀 Starting initial search...")

    # INITIAL SEARCH
    response = requests.post(
    search_url,
    auth=auth,
    json=query_body,
    headers={"Content-Type": "application/json"},
    timeout=30
)

    print("HTTP Status:", response.status_code)
    print("RAW Response (first 500 chars):")
    print(response.text[:500])


    data = response.json()

    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

    total_count = data["hits"]["total"]["value"] if "total" in data["hits"] else None
    print(f"⚡ Initial batch: {len(hits)} docs")

    all_docs = hits.copy()

    # tqdm progress bar
    pbar = tqdm(total=total_count, unit="docs") if total_count else tqdm(unit="docs")
    pbar.update(len(hits))

    # SCROLL LOOP
    while True:
        if not scroll_id:
            print("❌ No scroll_id returned. Stopping.")
            break

        response = requests.post(
            scroll_url,
            auth=auth,
            json={"scroll": "2m", "scroll_id": scroll_id},
            headers={"Content-Type": "application/json"}
        )

        data = response.json()
        hits = data["hits"]["hits"]

        if not hits:  # no more data
            print("✔ No more documents.")
            break

        all_docs.extend(hits)
        pbar.update(len(hits))

    pbar.close()
    print(f"🎉 Total retrieved: {len(all_docs)}")
    return all_docs



def safe_json(value):
    """Return JSON string for lists/dicts, or empty string."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value if value is not None else ""

# ==========================================
# CSV EXPORT (FIXED!)
# ==========================================
def save_to_csv(all_docs, filename="news_export.csv"):
    print("📦 Saving CSV...")

    fieldnames = [
        "_id",
        "Created_at",
        "Body",
        "Text",
        "Ticker",
        "Sentiment_score",
        "Sentiment_score_text",
        "fsai_Sentiment_score",
        "fsai_Sentiment_score_text",
        "fsai_Sentiment_label_text",
        "fsai_Sentiment_label",
        "url",
        "fsai_sentiment",
        "fsai_Targeted_Sentiment"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for doc in all_docs:
            src = doc.get("_source", {})

            writer.writerow({
                "_id": doc.get("_id"),
                "Created_at": src.get("Created_at"),
                "Body": src.get("Body"),
                "Text": src.get("Text"),
                "Ticker": src.get("Ticker"),
                "Sentiment_score": src.get("Sentiment_score"),
                "Sentiment_score_text": src.get("Sentiment_score_text"),
                "fsai_Sentiment_score": src.get("fsai_Sentiment_score"),
                "fsai_Sentiment_score_text": src.get("fsai_Sentiment_score_text"),
                "fsai_Sentiment_label_text": src.get("fsai_Sentiment_label_text"),
                "fsai_Sentiment_label": src.get("fsai_Sentiment_label"),
                "url": src.get("url"),
                "fsai_sentiment": safe_json(src.get("fsai_sentiment")),
                "fsai_Targeted_Sentiment": safe_json(src.get("fsai_Targeted_Sentiment"))
            })

    print(f"✅ CSV saved: {filename}")
# ==========================================
# RUN SCRIPT
# ==========================================
if __name__ == "__main__":
    docs = scroll_all_data()
    save_to_csv(docs)
