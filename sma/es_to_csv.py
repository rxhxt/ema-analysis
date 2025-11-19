import json
import pandas as pd
from pandas import json_normalize

def flatten_es_hits(json_path, output_csv="es_results_full.csv"):
    # Load JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract hits
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        print("No hits found in JSON file.")
        return

    # Flatten hits, including _source and metadata fields
    df = json_normalize(hits)

    # Optional: clean up complex nested lists/dicts into string form
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)

    # Save to CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"✅ CSV saved as: {output_csv}")
    print(f"📦 Total rows: {len(df)}")
    print("🧱 Columns exported:")
    for col in df.columns:
        print("  -", col)

# Run the conversion
if __name__ == "__main__":
    flatten_es_hits("../data/500_new_reddit.json", "../data/reddit_500_testing_v2.csv")
