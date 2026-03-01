import requests
import csv
import json
from tqdm import tqdm

USERNAME = "sastage"
PASSWORD = "Fsaistage#1"

index = "reddit"
host = "https://vpc-ssi-stg-sog3lhbiblkri6h6mqwun7jjde.ap-south-1.es.amazonaws.com"
search_url = f"{host}/{index}/_search?scroll=2m"
scroll_url = f"{host}/_search/scroll"
auth = (USERNAME, PASSWORD)

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


def scroll_all_data():
    print("🚀 Starting initial search...")

    MAX_DOCS = 1000

    resp = requests.post(
        search_url, auth=auth, json=query_body,
        headers={"Content-Type": "application/json"}, timeout=30
    )
    data = resp.json()

    scroll_id = data["_scroll_id"]
    hits = data["hits"]["hits"]

    all_docs = hits.copy()
    print(f"⚡ Initial docs: {len(hits)}")

    if len(all_docs) >= MAX_DOCS:
        return all_docs[:MAX_DOCS]

    pbar = tqdm(unit="docs")
    pbar.update(len(hits))

    while True:
        resp = requests.post(
            scroll_url,
            auth=auth,
            json={"scroll": "2m", "scroll_id": scroll_id},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        data = resp.json()
        hits = data["hits"]["hits"]

        if not hits:
            break

        all_docs.extend(hits)
        pbar.update(len(hits))

        if len(all_docs) >= MAX_DOCS:
            print(f"🛑 Reached limit of {MAX_DOCS} docs. Stopping early.")
            all_docs = all_docs[:MAX_DOCS]
            break

        scroll_id = data["_scroll_id"]

    pbar.close()
    print(f"🎉 Total retrieved: {len(all_docs)}")
    return all_docs

def save_to_csv(all_docs, filename="reddit_data.csv"):
    print("📦 Writing CSV...")

    fieldnames = [
        "_id",
        "Ticker",
        "Body",
        "Author",
        "url",
        "Created_at",
        "Score",
        "fsai_keywords",
        "fsai_sentiment",
        "fsai_targeted_sentiment",
        "fsai_entities",
        "fsai_svo",
        "fsai_concepts"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hit in all_docs:
            src = hit.get("_source", {})

            writer.writerow({
                "_id": hit.get("_id", ""),
                "Ticker": src.get("Ticker", ""),
                "Body": src.get("Body", ""),
                "Author": src.get("Author", ""),
                "url": src.get("url", ""),
                "Created_at": src.get("Created_at", ""),
                "Score": src.get("Score", ""),
                "fsai_keywords": json.dumps(src.get("fsai_keywords", [])),
                "fsai_sentiment": json.dumps(src.get("fsai_sentiment", {})),
                "fsai_targeted_sentiment": json.dumps(src.get("fsai_targeted_sentiment", [])),
                "fsai_entities": json.dumps(src.get("fsai_entities", [])),
                "fsai_svo": json.dumps(src.get("fsai_svo", [])),
                "fsai_concepts": json.dumps(src.get("fsai_concepts", []))
            })

    print(f"✅ CSV saved: {filename}")


if __name__ == "__main__":
    docs = scroll_all_data()
    save_to_csv(docs)
