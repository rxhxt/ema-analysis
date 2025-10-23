import pandas as pd
import os

# ==========================================================
# CONFIGURATION
# ==========================================================
SIGNALS_PATH = "output_signals"
OUTPUT_FILE = "all_signals_combined.csv"

# ==========================================================
# COMBINE ALL SIGNAL FILES (spec v1.1 compatible)
# ==========================================================
def combine_signals():
    all_data = []

    for file in os.listdir(SIGNALS_PATH):
        if not file.endswith("_signals.csv"):
            continue

        ticker = file.replace("_signals.csv", "")
        file_path = os.path.join(SIGNALS_PATH, file)

        df = pd.read_csv(file_path)
        df["ticker"] = ticker  # add ticker column

        # --- Expected new columns from improved pipeline ---
        expected_cols = [
            "ticker",
            "date",
            "SentimentScore",
            "PriceScore",
            "BuzzScore",
            "MomentumScore",
            "Signal",
            "Confidence",
            "SignalReason",
            "LowBuzzSkip",
            "sentiment_ema",
            "price_ema",
            "buzz_ema",
            "sent_trend14",
            "price_trend14",
            "buzz_trend14",
        ]

        # Add any missing columns as NaN (for backward compatibility)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

        # Reorder columns consistently
        df = df[expected_cols]

        all_data.append(df)

    if not all_data:
        print(f"⚠️ No signal files found in {SIGNALS_PATH}")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.sort_values(["ticker", "date"], inplace=True)

    # Save to CSV
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Combined {len(all_data)} tickers into {OUTPUT_FILE}")
    print(f"📊 Output columns: {list(combined_df.columns)}")

# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    combine_signals()
