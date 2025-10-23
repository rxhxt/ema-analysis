import pandas as pd
import numpy as np
import os

# ==========================================================
# CONFIGURATION (spec v1.1)
# ==========================================================
AV_PATH = "output_av"
NEWS_PATH = "output_news"
REDDIT_PATH = "output_reddit"
OUTPUT_PATH = "output_signals"

EMA_WINDOWS = {'sentiment': 3, 'buzz': 3, 'price': 7}      # Step 4
DELTA_LOOKBACKS = [1, 3, 7]                                # Step 5
DELTA_WEIGHTS = {1: 0.6, 3: 0.3, 7: 0.1}                   # Step 6

# Step 7 metric weights
METRIC_WEIGHTS = {'sentiment': 0.45, 'price': 0.35, 'buzz': 0.20}

# Step 9 thresholds
THRESHOLDS = {'strong': 0.10, 'moderate': 0.05}

# Step 3 low-buzz filters (absolute + relative to 30d median)
LOW_BUZZ_ABS_MIN = 8
LOW_BUZZ_REL_FRAC_OF_MEDIAN30 = 0.10  # 10%

# Step 8 context flatness epsilon (very small average 14d trend → “flat”)
CONTEXT_FLAT_EPS = 1e-3


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def ema(series: pd.Series, window: int) -> pd.Series:
    """EMA with alpha=2/(span+1), as per spec."""
    return series.ewm(span=window, adjust=False).mean()

def compute_deltas(series: pd.Series, lookbacks=DELTA_LOOKBACKS) -> dict:
    """Δn = EMA_t - EMA_{t-n} for n in {1,3,7}."""
    return {n: series - series.shift(n) for n in lookbacks}

def compute_metric_score(deltas: dict, delta_weights=DELTA_WEIGHTS) -> pd.Series:
    """MetricScore = 0.6*Δ1 + 0.3*Δ3 + 0.1*Δ7."""
    # Ensure we only combine keys present in deltas
    return sum(delta_weights[n] * deltas[n] for n in delta_weights.keys() if n in deltas)

def normalize_buzz(buzz_series: pd.Series) -> pd.Series:
    """Spec allows log1p OR relative-to-30d median. We default to log1p here."""
    return np.log1p(buzz_series)

def majority_sign(values) -> int:
    """Return 1 if majority positive, -1 if majority negative, else 0."""
    s = [1 if v > 0 else (-1 if v < 0 else 0) for v in values]
    pos, neg = s.count(1), s.count(-1)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0

def momentum_score_row(row) -> float:
    """Combine metric scores into MomentumScore (Step 7)."""
    return (
        METRIC_WEIGHTS['sentiment'] * row['SentimentScore'] +
        METRIC_WEIGHTS['price'] * row['PriceScore'] +
        METRIC_WEIGHTS['buzz'] * row['BuzzScore']
    )

def compute_signal_and_reason(row) -> tuple[str, str]:
    """
    Step 9 signal decision logic with metric agreement (2/3 and 3/3).
    Honors low-buzz skip via row['LowBuzzSkip'].
    """
    if row.get('LowBuzzSkip', False):
        return "SKIP_LOW_BUZZ", "Buzz below absolute or relative threshold"

    metrics = [row['SentimentScore'], row['PriceScore'], row['BuzzScore']]
    pos_count = sum(x > 0 for x in metrics)
    neg_count = sum(x < 0 for x in metrics)
    mscore = row['MomentumScore']

    if pos_count == 3 and mscore >= THRESHOLDS['strong']:
        return "STRONG BUY", "All 3 metrics positive and strong MomentumScore"
    if pos_count >= 2 and THRESHOLDS['moderate'] <= mscore < THRESHOLDS['strong']:
        return "BUY", "2 of 3 metrics positive and moderate MomentumScore"
    if neg_count == 3 and mscore <= -THRESHOLDS['strong']:
        return "STRONG SELL", "All 3 metrics negative and strong negative MomentumScore"
    if neg_count >= 2 and -THRESHOLDS['strong'] < mscore <= -THRESHOLDS['moderate']:
        return "SELL", "2 of 3 metrics negative and moderate negative MomentumScore"
    # Mixed or within noise band
    return "WATCH / HOLD", "Mixed metrics or |MomentumScore| < moderate threshold"

def compute_context_confidence(row) -> str:
    """
    Step 8: 14-day context check.
    - If average absolute 14d trend is ~flat → 'medium'
    - If momentum sign aligns with majority trend sign → 'high'
    - If majority trend is neutral → 'medium'
    - Else → 'low'
    """
    sent_t = row.get('sent_trend14', np.nan)
    buzz_t = row.get('buzz_trend14', np.nan)
    price_t = row.get('price_trend14', np.nan)
    mom = row.get('MomentumScore', np.nan)

    if np.isnan(sent_t) or np.isnan(buzz_t) or np.isnan(price_t) or np.isnan(mom):
        return "unknown"

    avg_abs_trend = np.nanmean([abs(sent_t), abs(buzz_t), abs(price_t)])
    maj = majority_sign([sent_t, buzz_t, price_t])

    if avg_abs_trend <= CONTEXT_FLAT_EPS:
        return "medium"  # strong score but flat backdrop
    if maj == 0:
        return "medium"
    if (mom > 0 and maj > 0) or (mom < 0 and maj < 0):
        return "high"
    return "low"

# ==========================================================
# CORE CALCULATION PIPELINE
# ==========================================================
def compute_momentum_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: ['date','sentiment','buzz','price']
    Implements Steps 2–9 of the spec on daily aggregates.
    """
    df = df.sort_values('date').copy()

    # ---- Step 2: Buzz normalization (kept separate from raw for filters) ----
    df['buzz_norm'] = normalize_buzz(df['buzz'])

    # ---- Step 4: EMA smoothing ----
    df['sentiment_ema'] = ema(df['sentiment'], EMA_WINDOWS['sentiment'])
    df['buzz_ema'] = ema(df['buzz_norm'], EMA_WINDOWS['buzz'])
    df['price_ema'] = ema(df['price'], EMA_WINDOWS['price'])

    # ---- Step 5 & 6: Deltas (1/3/7) and Metric Scores ----
    # Sentiment
    s_deltas = compute_deltas(df['sentiment_ema'])
    for n, series in s_deltas.items():
        df[f'sentiment_d{n}'] = series
    df['SentimentScore'] = compute_metric_score(s_deltas)

    # Buzz
    b_deltas = compute_deltas(df['buzz_ema'])
    for n, series in b_deltas.items():
        df[f'buzz_d{n}'] = series
    df['BuzzScore'] = compute_metric_score(b_deltas)

    # Price
    p_deltas = compute_deltas(df['price_ema'])
    for n, series in p_deltas.items():
        df[f'price_d{n}'] = series
    df['PriceScore'] = compute_metric_score(p_deltas)

    # ---- Step 7: MomentumScore ----
    df['MomentumScore'] = df.apply(momentum_score_row, axis=1)

    # ---- Step 8: 14-day context trends ----
    df['sent_trend14'] = df['sentiment_ema'] - df['sentiment_ema'].shift(14)
    df['buzz_trend14'] = df['buzz_ema'] - df['buzz_ema'].shift(14)
    df['price_trend14'] = df['price_ema'] - df['price_ema'].shift(14)

    # ---- Step 3: Low-buzz filters (absolute + relative) ----
    # Relative threshold uses RAW buzz median over last 30 days.
    med30 = df['buzz'].rolling(30, min_periods=1).median()
    abs_ok = df['buzz'] >= LOW_BUZZ_ABS_MIN
    rel_ok = df['buzz'] >= (LOW_BUZZ_REL_FRAC_OF_MEDIAN30 * med30)
    df['LowBuzzSkip'] = ~(abs_ok & rel_ok)

    # ---- Step 9: Signal with metric agreement + thresholds ----
    out = []
    for _, row in df.iterrows():
        signal, reason = compute_signal_and_reason(row)
        out.append((signal, reason))
    df['Signal'] = [s for s, _ in out]
    df['SignalReason'] = [r for _, r in out]

    # ---- Context confidence (Step 8) and final confidence label ----
    df['ContextConfidence'] = df.apply(compute_context_confidence, axis=1)

    # Optional “rolled up” Confidence combining signal class + context:
    def final_conf(sig: str, ctx: str) -> str:
        if ctx == "unknown":
            return "unknown"
        if sig in {"STRONG BUY", "STRONG SELL"}:
            return "high" if ctx == "high" else ("medium" if ctx in {"medium", "low"} else "unknown")
        if sig in {"BUY", "SELL"}:
            return ctx if ctx in {"high", "medium"} else "low"
        if sig in {"WATCH / HOLD", "SKIP_LOW_BUZZ"}:
            return "low"
        return "low"

    df['Confidence'] = [final_conf(s, c) for s, c in zip(df['Signal'], df['ContextConfidence'])]

    # Final output table
    return df[[
        'date',
        'SentimentScore', 'PriceScore', 'BuzzScore',
        'MomentumScore', 'Signal', 'Confidence',
        'SignalReason', 'LowBuzzSkip',
        # Handy diagnostics (optional to keep/export)
        'sentiment_ema', 'price_ema', 'buzz_ema',
        'sent_trend14', 'price_trend14', 'buzz_trend14'
    ]]

# ==========================================================
# IO / MERGE PIPELINE (kept compatible with your folder layout)
# ==========================================================
def process_all_tickers():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for file in os.listdir(AV_PATH):
        if not file.endswith("_data_av.csv"):
            continue

        ticker = file.replace("_data_av.csv", "")
        av_file = os.path.join(AV_PATH, file)
        news_file = os.path.join(NEWS_PATH, f"{ticker}_data_news.csv")
        reddit_file = os.path.join(REDDIT_PATH, f"{ticker}_data_reddit.csv")

        if not os.path.exists(news_file) or not os.path.exists(reddit_file):
            print(f"⚠️ Skipping {ticker}: missing news or reddit data")
            continue

        # Load data
        av_df = pd.read_csv(av_file)
        news_df = pd.read_csv(news_file)
        reddit_df = pd.read_csv(reddit_file)

        # Rename expected columns
        av_df = av_df.rename(columns={'avg_price': 'price'})
        news_df = news_df.rename(columns={
            'daily_volume': 'buzz_news',
            'avg_fsai_Sentiment_score': 'sent_news'
        })
        reddit_df = reddit_df.rename(columns={
            'daily_volume': 'buzz_reddit',
            'avg_fsai_Sentiment_score': 'sent_reddit'
        })

        # Merge sources on date
        df = pd.merge(
            news_df[['date', 'buzz_news', 'sent_news']],
            reddit_df[['date', 'buzz_reddit', 'sent_reddit']],
            on='date', how='outer'
        )

        # Aggregate daily sentiment and buzz (simple mean for sentiment, mean for buzz)
        df['sentiment'] = (df['sent_news'].fillna(0) + df['sent_reddit'].fillna(0)) / 2.0
        df['buzz'] = (df['buzz_news'].fillna(0) + df['buzz_reddit'].fillna(0)) / 2.0

        # Merge price data (inner to ensure a price exists)
        df = pd.merge(df, av_df[['date', 'price']], on='date', how='inner')

        # Persist intermediate combined file for debugging
        output_mid_file = os.path.join(OUTPUT_PATH, f"{ticker}_combined.csv")
        df.to_csv(output_mid_file, index=False)

        # Clean
        df = df.dropna(subset=['price'])
        # Ensure correct dtypes
        df['date'] = pd.to_datetime(df['date'])
        for col in ['sentiment', 'buzz', 'price']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute momentum alerts (spec compliant)
        result = compute_momentum_alerts(df)

        # Save results
        output_file = os.path.join(OUTPUT_PATH, f"{ticker}_signals.csv")
        result.to_csv(output_file, index=False)
        print(f"✅ Processed {ticker}: saved to {output_file}")

# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    process_all_tickers()
