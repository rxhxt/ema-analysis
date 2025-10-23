# Momentum Alert Output Columns

| Column Name         | Description |
|---------------------|-------------|
| **date** | Analysis day (end of 6 AM ET → 6 AM ET window). |
| **SentimentScore** | Composite score for sentiment (0.6·Δ₁ + 0.3·Δ₃ + 0.1·Δ₇ from 3-day EMA). |
| **PriceScore** | Composite score for price (0.6·Δ₁ + 0.3·Δ₃ + 0.1·Δ₇ from 7-day EMA). |
| **BuzzScore** | Composite score for buzz (0.6·Δ₁ + 0.3·Δ₃ + 0.1·Δ₇ from 3-day EMA). |
| **MomentumScore** | Weighted combination: 0.45·Sentiment + 0.35·Price + 0.20·Buzz. |
| **Signal** | Final momentum signal category:<br>• STRONG BUY / BUY / SELL / STRONG SELL / WATCH / HOLD / SKIP_LOW_BUZZ |
| **Confidence** | Combined confidence level based on 14-day context:<br>• high / medium / low / unknown |
| **SignalReason** | Short text explanation of why the signal was assigned (e.g. “All 3 metrics positive and strong MomentumScore”). |
| **LowBuzzSkip** | Boolean flag — `True` if day skipped due to buzz < 8 or < 10 % of 30-day median. |
| **sentiment_ema** | 3-day EMA of sentiment (smoothed tone). |
| **price_ema** | 7-day EMA of closing price (smoothed price trend). |
| **buzz_ema** | 3-day EMA of normalized buzz (smoothed attention trend). |
| **sent_trend14** | 14-day sentiment EMA change (context trend). |
| **price_trend14** | 14-day price EMA change (context trend). |
| **buzz_trend14** | 14-day buzz EMA change (context trend). |
| **ContextConfidence** | Confidence derived purely from 14-day context alignment (before combining with signal class). |

> ⚙️ *Tip:* You can safely drop the diagnostic EMA and trend columns when exporting only the headline signal output:
> `['date', 'SentimentScore', 'PriceScore', 'BuzzScore', 'MomentumScore', 'Signal', 'Confidence']`.
