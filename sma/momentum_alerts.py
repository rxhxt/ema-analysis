import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path

# Check for Excel support
try:
    import xlsxwriter
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False
    print("WARNING: xlsxwriter not installed. Excel output will be skipped.")
    print("Install with: pip install xlsxwriter")

# ==========================================================
# CONFIGURATION (from your pandas script)
# ==========================================================
EMA_WINDOWS = {'sentiment': 3, 'buzz': 3, 'price': 7}
DELTA_LOOKBACKS = [1, 3, 7]
DELTA_WEIGHTS = {1: 0.6, 3: 0.3, 7: 0.1}
METRIC_WEIGHTS = {'sentiment': 0.45, 'price': 0.35, 'buzz': 0.20}
THRESHOLDS = {'strong': 0.10, 'moderate': 0.05}
LOW_BUZZ_ABS_MIN = 8
LOW_BUZZ_REL_FRAC_OF_MEDIAN30 = 0.10
CONTEXT_FLAT_EPS = 1e-3

def calculate_metric_score_vectorized(df: pl.DataFrame, ema_col: str, lookbacks: list, weights: dict) -> pl.Expr:
    """
    Calculate metric score using delta method: MetricScore = 0.6*Δ1 + 0.3*Δ3 + 0.1*Δ7
    where Δn = EMA_t - EMA_{t-n}
    """
    score = pl.lit(0.0)
    
    for n in lookbacks:
        if n in weights:
            delta = pl.col(ema_col) - pl.col(ema_col).shift(n)
            score = score + (delta.fill_null(0) * weights[n])
    
    return score

def majority_sign_vectorized(col1: pl.Expr, col2: pl.Expr, col3: pl.Expr) -> pl.Expr:
    """Return 1 if majority positive, -1 if majority negative, else 0"""
    pos_count = (
        (col1 > 0).cast(pl.Int32) + 
        (col2 > 0).cast(pl.Int32) + 
        (col3 > 0).cast(pl.Int32)
    )
    neg_count = (
        (col1 < 0).cast(pl.Int32) + 
        (col2 < 0).cast(pl.Int32) + 
        (col3 < 0).cast(pl.Int32)
    )
    
    return (
        pl.when(pos_count > neg_count).then(pl.lit(1))
        .when(neg_count > pos_count).then(pl.lit(-1))
        .otherwise(pl.lit(0))
    )

def compute_context_confidence_vectorized(
    sent_trend14: pl.Expr, 
    buzz_trend14: pl.Expr, 
    price_trend14: pl.Expr, 
    momentum_score: pl.Expr
) -> pl.Expr:
    """
    Step 8: 14-day context check vectorized
    """
    avg_abs_trend = (
        sent_trend14.abs() + buzz_trend14.abs() + price_trend14.abs()
    ) / 3.0
    
    maj_sign = majority_sign_vectorized(sent_trend14, buzz_trend14, price_trend14)
    mom_sign = pl.when(momentum_score > 0).then(pl.lit(1)).when(momentum_score < 0).then(pl.lit(-1)).otherwise(pl.lit(0))
    
    return (
        pl.when(avg_abs_trend <= CONTEXT_FLAT_EPS).then(pl.lit("medium"))
        .when(maj_sign == 0).then(pl.lit("medium"))
        .when((mom_sign > 0) & (maj_sign > 0)).then(pl.lit("high"))
        .when((mom_sign < 0) & (maj_sign < 0)).then(pl.lit("high"))
        .otherwise(pl.lit("low"))
    )

def generate_signal_vectorized(
    sentiment_score: pl.Expr, 
    price_score: pl.Expr, 
    buzz_score: pl.Expr,
    momentum_score: pl.Expr,
    low_buzz_skip: pl.Expr
) -> pl.Expr:
    """
    Generate trading signals with metric agreement logic (2/3 and 3/3)
    """
    pos_count = (
        (sentiment_score > 0).cast(pl.Int32) + 
        (price_score > 0).cast(pl.Int32) + 
        (buzz_score > 0).cast(pl.Int32)
    )
    neg_count = (
        (sentiment_score < 0).cast(pl.Int32) + 
        (price_score < 0).cast(pl.Int32) + 
        (buzz_score < 0).cast(pl.Int32)
    )
    
    return (
        pl.when(low_buzz_skip).then(pl.lit('SKIP_LOW_BUZZ'))
        .when((pos_count == 3) & (momentum_score >= THRESHOLDS['strong'])).then(pl.lit('STRONG BUY'))
        .when((pos_count >= 2) & (momentum_score >= THRESHOLDS['moderate']) & (momentum_score < THRESHOLDS['strong'])).then(pl.lit('BUY'))
        .when((neg_count == 3) & (momentum_score <= -THRESHOLDS['strong'])).then(pl.lit('STRONG SELL'))
        .when((neg_count >= 2) & (momentum_score > -THRESHOLDS['strong']) & (momentum_score <= -THRESHOLDS['moderate'])).then(pl.lit('SELL'))
        .otherwise(pl.lit('WATCH / HOLD'))
        .alias('signal')
    )

def process_stock_data(stock_ticker: str, reddit_file: str = None, news_file: str = None, av_file: str = None) -> pl.DataFrame:
    """Process data for a single stock - returns DataFrame"""
    
    # Load market data (avg_vol, avg_price)
    df_market = None
    if av_file and Path(av_file).exists():
        df_market = pl.read_csv(av_file).with_columns(
            pl.col('date').str.to_datetime()
        ).sort('date')
        print(f"  [{stock_ticker}] Loaded market data: {len(df_market)} rows")
    
    # Load reddit data (daily_volume, sentiment scores)
    df_reddit = None
    if reddit_file and Path(reddit_file).exists():
        df_reddit = pl.read_csv(reddit_file).with_columns(
            pl.col('date').str.to_datetime()
        ).sort('date').rename({
            'daily_volume': 'reddit_volume',
            'avg_fsai_Sentiment_score': 'reddit_sentiment_score'
        })
        print(f"  [{stock_ticker}] Loaded reddit data: {len(df_reddit)} rows")
    
    # Load news data (daily_volume, sentiment scores)
    df_news = None
    if news_file and Path(news_file).exists():
        df_news = pl.read_csv(news_file).with_columns(
            pl.col('date').str.to_datetime()
        ).sort('date').rename({
            'daily_volume': 'news_volume',
            'avg_fsai_Sentiment_score': 'news_sentiment_score'
        })
        print(f"  [{stock_ticker}] Loaded news data: {len(df_news)} rows")
    
    # Must have at least market data
    if df_market is None:
        print(f"  [{stock_ticker}] ERROR: No market data file found")
        return None
    
    # Start with market data as base
    df = df_market
    
    # Join reddit data if available
    if df_reddit is not None:
        df = df.join(df_reddit, on='date', how='left')
    
    # Join news data if available
    if df_news is not None:
        df = df.join(df_news, on='date', how='left')
    
    # Calculate combined sentiment (average of reddit and news)
    sentiment_cols = []
    if 'reddit_sentiment_score' in df.columns:
        sentiment_cols.append('reddit_sentiment_score')
    if 'news_sentiment_score' in df.columns:
        sentiment_cols.append('news_sentiment_score')
    
    if sentiment_cols:
        if len(sentiment_cols) == 1:
            df = df.with_columns(pl.col(sentiment_cols[0]).alias('combined_sentiment'))
        else:
            df = df.with_columns(
                pl.concat_list([pl.col(c).fill_null(0) for c in sentiment_cols])
                .list.mean()
                .alias('combined_sentiment')
            )
    else:
        print(f"  [{stock_ticker}] WARNING: No sentiment data, using default 0.5")
        df = df.with_columns(pl.lit(0.5).alias('combined_sentiment'))
    
    # Calculate combined buzz (average of reddit and news volume, NOT including market volume)
    buzz_cols = []
    if 'reddit_volume' in df.columns:
        buzz_cols.append('reddit_volume')
    if 'news_volume' in df.columns:
        buzz_cols.append('news_volume')
    
    if buzz_cols:
        if len(buzz_cols) == 1:
            df = df.with_columns(pl.col(buzz_cols[0]).alias('combined_buzz'))
        else:
            df = df.with_columns(
                pl.concat_list([pl.col(c).fill_null(0) for c in buzz_cols])
                .list.mean()
                .alias('combined_buzz')
            )
    else:
        print(f"  [{stock_ticker}] WARNING: No buzz data, using default 1000")
        df = df.with_columns(pl.lit(1000.0).alias('combined_buzz'))
    
    # Use actual price from market data
    if 'avg_price' in df.columns:
        df = df.with_columns(pl.col('avg_price').alias('price'))
    else:
        print(f"  [{stock_ticker}] WARNING: No price data, generating dummy prices")
        np.random.seed(hash(stock_ticker) % 2**32)
        prices = 100 + np.cumsum(np.random.randn(len(df)) * 2)
        df = df.with_columns(pl.lit(prices).alias('price'))
    
    # Fill nulls with forward/backward fill strategy
    df = df.with_columns([
        pl.col('combined_sentiment').fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0.5),
        pl.col('combined_buzz').fill_null(strategy='forward').fill_null(strategy='backward').fill_null(1000),
        pl.col('price').fill_null(strategy='forward').fill_null(strategy='backward').fill_null(100)
    ])
    
    # Step 2: Normalize buzz using log1p (as per spec)
    df = df.with_columns([
        pl.col('combined_buzz').log1p().alias('buzz_normalized')
    ])
    
    # Step 4: Calculate EMAs with SPEC windows (3 for sentiment/buzz, 7 for price)
    df = df.with_columns([
        pl.col('combined_sentiment').ewm_mean(span=EMA_WINDOWS['sentiment'], adjust=False).alias('sentiment_ema'),
        pl.col('buzz_normalized').ewm_mean(span=EMA_WINDOWS['buzz'], adjust=False).alias('buzz_ema'),
        pl.col('price').ewm_mean(span=EMA_WINDOWS['price'], adjust=False).alias('price_ema')
    ])
    
    # Step 5 & 6: Calculate metric scores using delta method
    df = df.with_columns([
        calculate_metric_score_vectorized(df, 'sentiment_ema', DELTA_LOOKBACKS, DELTA_WEIGHTS).alias('sentiment_score'),
        calculate_metric_score_vectorized(df, 'buzz_ema', DELTA_LOOKBACKS, DELTA_WEIGHTS).alias('buzz_score'),
        calculate_metric_score_vectorized(df, 'price_ema', DELTA_LOOKBACKS, DELTA_WEIGHTS).alias('price_score')
    ])
    
    # Step 7: Calculate MomentumScore
    df = df.with_columns([
        (pl.col('sentiment_score') * METRIC_WEIGHTS['sentiment'] + 
         pl.col('price_score') * METRIC_WEIGHTS['price'] + 
         pl.col('buzz_score') * METRIC_WEIGHTS['buzz']).alias('momentum_score')
    ])
    
    # Step 8: 14-day context trends
    df = df.with_columns([
        (pl.col('sentiment_ema') - pl.col('sentiment_ema').shift(14)).alias('sent_trend14'),
        (pl.col('buzz_ema') - pl.col('buzz_ema').shift(14)).alias('buzz_trend14'),
        (pl.col('price_ema') - pl.col('price_ema').shift(14)).alias('price_trend14')
    ])
    
    # Step 3: Low-buzz filters (absolute + relative to 30d median)
    df = df.with_columns([
        pl.col('combined_buzz').rolling_median(window_size=30, min_periods=1).alias('buzz_median_30d')
    ])
    
    df = df.with_columns([
        (
            (pl.col('combined_buzz') < LOW_BUZZ_ABS_MIN) |
            (pl.col('combined_buzz') < (LOW_BUZZ_REL_FRAC_OF_MEDIAN30 * pl.col('buzz_median_30d')))
        ).alias('low_buzz_skip')
    ])
    
    # Step 9: Generate signals with metric agreement
    df = df.with_columns([
        generate_signal_vectorized(
            pl.col('sentiment_score'),
            pl.col('price_score'),
            pl.col('buzz_score'),
            pl.col('momentum_score'),
            pl.col('low_buzz_skip')
        )
    ])
    
    # Calculate context confidence
    df = df.with_columns([
        compute_context_confidence_vectorized(
            pl.col('sent_trend14').fill_null(0),
            pl.col('buzz_trend14').fill_null(0),
            pl.col('price_trend14').fill_null(0),
            pl.col('momentum_score')
        ).alias('context_confidence')
    ])
    
    # Final confidence combining signal + context
    df = df.with_columns([
        pl.when(pl.col('signal').is_in(['STRONG BUY', 'STRONG SELL']))
        .then(
            pl.when(pl.col('context_confidence') == 'high').then(pl.lit('high'))
            .otherwise(pl.lit('medium'))
        )
        .when(pl.col('signal').is_in(['BUY', 'SELL']))
        .then(pl.col('context_confidence'))
        .otherwise(pl.lit('low'))
        .alias('confidence')
    ])
    
    # Add ticker column
    df = df.with_columns([
        pl.lit(stock_ticker).alias('ticker')
    ])
    
    print(f"  [{stock_ticker}] Processing complete: {len(df)} days, {df['signal'].value_counts().height} unique signals")
    
    return df

def write_output_1_csv(df: pl.DataFrame, output_dir: Path):
    """OUTPUT 1: Day-wise alerts as CSV"""
    output_file = output_dir / 'output_1_daywise_alerts.csv'
    
    # Create pivot table structure
    pivot_data = []
    
    for date in df['date'].unique().sort():
        date_df = df.filter(pl.col('date') == date)
        
        row = {
            'date': date,
            'strong_buy': ', '.join(date_df.filter(pl.col('signal') == 'STRONG BUY')['ticker'].to_list()),
            'buy': ', '.join(date_df.filter(pl.col('signal') == 'BUY')['ticker'].to_list()),
            'watch_hold': ', '.join(date_df.filter(pl.col('signal') == 'WATCH / HOLD')['ticker'].to_list()),
            'sell': ', '.join(date_df.filter(pl.col('signal') == 'SELL')['ticker'].to_list()),
            'strong_sell': ', '.join(date_df.filter(pl.col('signal') == 'STRONG SELL')['ticker'].to_list()),
            'skip_low_buzz': ', '.join(date_df.filter(pl.col('signal') == 'SKIP_LOW_BUZZ')['ticker'].to_list())
        }
        pivot_data.append(row)
    
    pivot_df = pl.DataFrame(pivot_data)
    pivot_df.write_csv(output_file)
    
    print(f"✓ Output 1 written to: {output_file}")

def write_output_2_excel(df: pl.DataFrame, output_dir: Path):
    """OUTPUT 2: Detailed metadata as Excel"""
    if not EXCEL_SUPPORT:
        print("⚠ Skipping Excel output (xlsxwriter not installed)")
        return
    
    output_file = output_dir / 'output_2_detailed_metadata.xlsx'
    
    # Select relevant columns for output
    output_cols = [
        'date', 'ticker', 'signal', 'confidence',
        'sentiment_ema', 'buzz_ema', 'price_ema',
        'sentiment_score', 'buzz_score', 'price_score', 
        'momentum_score',
        'sent_trend14', 'buzz_trend14', 'price_trend14',
        'context_confidence'
    ]
    
    detailed_df = df.select(output_cols).sort(['date', 'ticker'])
    detailed_df.write_excel(output_file)
    
    print(f"✓ Output 2 written to: {output_file}")

def write_output_3_csv(df: pl.DataFrame, output_dir: Path):
    """OUTPUT 3: Latest day summary as CSV"""
    output_file = output_dir / 'output_3_latest_day_summary.csv'
    
    # Get latest date data
    latest_date = df['date'].max()
    latest_df = df.filter(pl.col('date') == latest_date).sort('momentum_score', descending=True)
    
    # Select key columns
    summary_cols = [
        'ticker', 'signal', 'confidence',
        'sentiment_score', 'buzz_score', 'price_score', 'momentum_score'
    ]
    
    summary_df = latest_df.select(summary_cols)
    summary_df.write_csv(output_file)
    
    print(f"✓ Output 3 written to: {output_file}")
    
    # Also create a statistics summary
    stats_file = output_dir / 'output_3_signal_statistics.csv'
    
    signal_stats = (
        latest_df
        .group_by('signal')
        .agg([
            pl.len().alias('count'),
            pl.col('momentum_score').mean().alias('avg_score'),
            pl.col('momentum_score').min().alias('min_score'),
            pl.col('momentum_score').max().alias('max_score')
        ])
        .sort('signal')
    )
    
    signal_stats.write_csv(stats_file)
    print(f"✓ Signal statistics written to: {stats_file}")

def write_combined_excel(df: pl.DataFrame, output_dir: Path):
    """Write all outputs to a single Excel file with multiple sheets"""
    if not EXCEL_SUPPORT:
        print("⚠ Skipping combined Excel output (xlsxwriter not installed)")
        return
    
    output_file = output_dir / 'momentum_alerts_complete.xlsx'
    
    import xlsxwriter
    
    workbook = xlsxwriter.Workbook(output_file)
    
    # Sheet 1: Day-wise alerts (pivoted)
    pivot_data = []
    for date in df['date'].unique().sort():
        date_df = df.filter(pl.col('date') == date)
        row = {
            'date': date,
            'strong_buy': ', '.join(date_df.filter(pl.col('signal') == 'STRONG BUY')['ticker'].to_list()),
            'buy': ', '.join(date_df.filter(pl.col('signal') == 'BUY')['ticker'].to_list()),
            'watch_hold': ', '.join(date_df.filter(pl.col('signal') == 'WATCH / HOLD')['ticker'].to_list()),
            'sell': ', '.join(date_df.filter(pl.col('signal') == 'SELL')['ticker'].to_list()),
            'strong_sell': ', '.join(date_df.filter(pl.col('signal') == 'STRONG SELL')['ticker'].to_list()),
            'skip_low_buzz': ', '.join(date_df.filter(pl.col('signal') == 'SKIP_LOW_BUZZ')['ticker'].to_list())
        }
        pivot_data.append(row)
    
    pivot_df = pl.DataFrame(pivot_data)
    
    worksheet1 = workbook.add_worksheet('Daywise_Alerts')
    _write_df_to_worksheet(pivot_df, worksheet1, workbook)
    
    # Sheet 2: Detailed metadata
    output_cols = [
        'date', 'ticker', 'signal', 'confidence',
        'sentiment_ema', 'buzz_ema', 'price_ema',
        'sentiment_score', 'buzz_score', 'price_score',
        'momentum_score',
        'sent_trend14', 'buzz_trend14', 'price_trend14',
        'context_confidence'
    ]
    detailed_df = df.select(output_cols).sort(['date', 'ticker'])
    
    worksheet2 = workbook.add_worksheet('Detailed_Metadata')
    _write_df_to_worksheet(detailed_df, worksheet2, workbook)
    
    # Sheet 3: Latest day summary
    latest_date = df['date'].max()
    latest_df = df.filter(pl.col('date') == latest_date).sort('momentum_score', descending=True)
    summary_cols = ['ticker', 'signal', 'confidence', 'sentiment_score', 'buzz_score', 'price_score', 'momentum_score']
    summary_df = latest_df.select(summary_cols)
    
    worksheet3 = workbook.add_worksheet('Latest_Day_Summary')
    _write_df_to_worksheet(summary_df, worksheet3, workbook)
    
    # Sheet 4: Signal statistics
    signal_stats = (
        latest_df
        .group_by('signal')
        .agg([
            pl.len().alias('count'),
            pl.col('momentum_score').mean().alias('avg_score'),
            pl.col('momentum_score').min().alias('min_score'),
            pl.col('momentum_score').max().alias('max_score')
        ])
        .sort('signal')
    )
    
    worksheet4 = workbook.add_worksheet('Signal_Statistics')
    _write_df_to_worksheet(signal_stats, worksheet4, workbook)
    
    workbook.close()
    
    print(f"✓ Combined Excel file written to: {output_file}")

def _write_df_to_worksheet(df: pl.DataFrame, worksheet, workbook):
    """Helper to write Polars DataFrame to xlsxwriter worksheet"""
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D9E1F2',
        'border': 1
    })
    
    # Write headers
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, col_name, header_format)
    
    # Write data
    for row_idx, row in enumerate(df.iter_rows(), start=1):
        for col_idx, value in enumerate(row):
            if hasattr(value, 'strftime'):
                value = value.strftime('%Y-%m-%d')
            worksheet.write(row_idx, col_idx, value)
    
    # Auto-fit columns
    for col_idx, col_name in enumerate(df.columns):
        max_len = max(len(str(col_name)), 
                     max((len(str(val)) for val in df[col_name].to_list()), default=0))
        worksheet.set_column(col_idx, col_idx, min(max_len + 2, 50))

def main():
    """Main function"""
    print("\n" + "=" * 120)
    print("MOMENTUM ALERT SYSTEM - SPEC-COMPLIANT POLARS VERSION")
    print("=" * 120)
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Stock configuration
    stocks = {
        'AMD': {'reddit': 'AMD_data_reddit.csv', 'news': 'AMD_data_news.csv', 'av': 'AMD_data_av.csv'},
        'TSLA': {'reddit': 'TSLA_data_reddit.csv', 'news': 'TSLA_data_news.csv', 'av': 'TSLA_data_av.csv'},
        'COIN': {'reddit': 'COIN_data_reddit.csv', 'news': 'COIN_data_news.csv', 'av': 'COIN_data_av.csv'},
        'COST': {'reddit': 'COST_data_reddit.csv', 'news': 'COST_data_news.csv', 'av': 'COST_data_av.csv'},
        'AAPL': {'reddit': 'AAPL_data_reddit.csv', 'news': 'AAPL_data_news.csv', 'av': 'AAPL_data_av.csv'},
        'SNOW': {'reddit': 'SNOW_data_reddit.csv', 'news': 'SNOW_data_news.csv', 'av': 'SNOW_data_av.csv'},
        'META': {'reddit': 'META_data_reddit.csv', 'news': 'META_data_news.csv', 'av': 'META_data_av.csv'},
        'MSFT': {'reddit': 'MSFT_data_reddit.csv', 'news': 'MSFT_data_news.csv', 'av': 'MSFT_data_av.csv'},
        'NVDA': {'reddit': 'NVDA_data_reddit.csv', 'news': 'NVDA_data_news.csv', 'av': 'NVDA_data_av.csv'},
        'PLTR': {'reddit': 'PLTR_data_reddit.csv', 'news': 'PLTR_data_news.csv', 'av': 'PLTR_data_av.csv'},
        'GS': {'reddit': 'GS_data_reddit.csv', 'news': 'GS_data_news.csv', 'av': 'GS_data_av.csv'}
    }
    
    # Process all stocks
    all_dfs = []
    for ticker, files in stocks.items():
        print(f"\nProcessing {ticker}...")
        df = process_stock_data(
            stock_ticker=ticker,
            reddit_file=files.get('reddit'),
            news_file=files.get('news'),
            av_file=files.get('av')
        )
        if df is not None:
            all_dfs.append(df)
        else:
            print(f"  [{ticker}] ✗ Skipped due to errors")
    
    # Combine all dataframes
    if not all_dfs:
        print("\n❌ No data to process!")
        return
    
    combined_df = pl.concat(all_dfs, how='vertical')
    
    print(f"\n{'=' * 120}")
    print(f"TOTAL DATA PROCESSED: {len(combined_df)} rows across {len(all_dfs)} stocks")
    print(f"{'=' * 120}")
    
    # Write outputs
    print("\nGenerating outputs...")
    
    write_output_1_csv(combined_df, output_dir)
    write_output_2_excel(combined_df, output_dir)
    write_output_3_csv(combined_df, output_dir)
    write_combined_excel(combined_df, output_dir)
    
    print("\n" + "=" * 120)
    print("✅ ANALYSIS COMPLETE - Check 'outputs' folder")
    print("=" * 120)

if __name__ == "__main__":
    main()