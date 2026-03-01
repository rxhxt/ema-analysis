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
# QUERY BODY (all fields)
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
    "size": 500
}

# ==========================================
# SCROLL FUNCTION
# ==========================================
def scroll_all_data(max_docs=1000):
    print("🚀 Starting initial search...")

    response = requests.post(
        search_url,
        auth=auth,
        json=query_body,
        headers={"Content-Type": "application/json"},
        timeout=30
    )

    data = response.json()

    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

    total = data["hits"]["total"]["value"]
    print(f"⚡ Initial docs: {len(hits)}")

    all_docs = hits.copy()
    
    # If we already have enough docs, return early
    if len(all_docs) >= max_docs:
        print(f"🛑 Reached limit of {max_docs} docs.")
        return all_docs[:max_docs]
    
    pbar = tqdm(total=min(total, max_docs), unit="docs")
    pbar.update(len(hits))

    while len(all_docs) < max_docs:
        response = requests.post(
            scroll_url,
            auth=auth,
            json={"scroll": "2m", "scroll_id": scroll_id},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        data = response.json()
        hits = data["hits"]["hits"]

        if not hits:
            break

        all_docs.extend(hits)
        pbar.update(min(len(hits), max_docs - len(all_docs) + len(hits)))
        
        if len(all_docs) >= max_docs:
            print(f"🛑 Reached limit of {max_docs} docs. Stopping early.")
            all_docs = all_docs[:max_docs]
            break
        
        scroll_id = data.get("_scroll_id")

    pbar.close()
    print(f"🎉 Total retrieved: {len(all_docs)}")
    return all_docs

# ==========================================
# SAFE JSON SERIALIZER
# ==========================================
def safe_json(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else value

# ==========================================
# CSV EXPORT
# ==========================================
def save_to_csv(all_docs, filename="news_fsai_export.csv"):
    print("📦 Saving CSV...")

    fieldnames = [
        "_id",
        "Ticker",
        "Body",
        "Text",
        "Author",
        "Host",
        "url",
        "Created_at",
        "market_time_news",
        "fsai_Sentiment_score_text",
        "fsai_Sentiment_label_text",
        "fsai_Targeted_Sentiment",
        "fsai_Keyword",
        "fsai_SVO",
        "fsai_Company",
        "fsai_Person",
        "fsai_Concept",
        "fsai_Categories"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hit in all_docs:
            src = hit.get("_source", {})

            writer.writerow({
                "_id": hit.get("_id"),
                "Ticker": src.get("Ticker", ""),
                "Body": src.get("Body", ""),
                "Text": src.get("Text", ""),
                "Author": src.get("Author", ""),
                "Host": src.get("Host", ""),
                "url": src.get("url", ""),
                "Created_at": src.get("Created_at", ""),
                "market_time_news": src.get("market_time_news", ""),

                # sentiment fields
                "fsai_Sentiment_score_text": src.get("fsai_Sentiment_score_text", ""),
                "fsai_Sentiment_label_text": src.get("fsai_Sentiment_label_text", ""),

                # complex fields serialized safely
                "fsai_Targeted_Sentiment": safe_json(src.get("fsai_Targeted_Sentiment")),
                "fsai_Keyword": safe_json(src.get("fsai_Keyword")),
                "fsai_SVO": safe_json(src.get("fsai_SVO")),
                "fsai_Company": safe_json(src.get("fsai_Company")),
                "fsai_Person": safe_json(src.get("fsai_Person")),
                "fsai_Concept": safe_json(src.get("fsai_Concept")),
                "fsai_Categories": safe_json(src.get("fsai_Categories"))
            })

    print(f"✅ CSV saved: {filename}")


# ==========================================
# RUN SCRIPT
# ==========================================
if __name__ == "__main__":
    docs = scroll_all_data(max_docs=1000)
    save_to_csv(docs)
