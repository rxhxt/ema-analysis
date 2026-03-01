import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
import altair as alt

# ----------------------------------------
# 🔧 Configuration
# ----------------------------------------
API_URL = "https://25becp9ef3.execute-api.ap-south-1.amazonaws.com/dev/ema-alerts"
ENV_KEY = "dev"  # or "prod" / "stable"
BATCH_SIZE = 20  # Tickers per API call

# All available tickers (143 stocks across 11 sectors)
ALL_TICKERS = ["META", "DIS", "GOOGL", "NFLX", "ASTS", "RBLX", "RDDT", "SPOT", "F", "TSLA", "UBER", "MAR", "AMZN", "GME", "DKNG", "LCID", "LULU", "NKE", "RIVN", "GM", "SBUX", "MCD", "CMG", "DPZ", "DASH", "TTM", "BABA", "MMYT", "PG", "WMT", "COST", "KO", "PEP", "PM", "CELH", "TGT", "XOM", "CVX", "SHEL", "NFE", "COP", "BAC", "JPM", "V", "PYPL", "XYZ", "GS", "COIN", "HOOD", "MA", "NU", "SOFI", "C", "BK", "HDB", "IBN", "PFE", "NVAX", "JNJ", "CVS", "MRNA", "UNH", "VKTX", "LLY", "SNDX", "RDY", "BA", "CAT", "FDX", "MMM", "HON", "ETN", "UNP", "RTX", "GE", "MSFT", "INTC", "AAPL", "AMD", "NVDA", "AVGO", "CRM", "CRWD", "MSTR", "PLTR", "SNOW", "APP", "MU", "SMCI", "ZETA", "U", "DELL", "ORCL", "ADBE", "ZS", "SHOP", "TSM", "INTU", "DDOG", "DOCU", "ENPH", "MRVL", "WIT", "INFY", "LIN", "O", "NEE", "DUK"]

st.set_page_config(page_title="Momentum Signal Dashboard", layout="wide")

# Custom CSS for blue ticker pills
st.markdown("""
<style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #2E86DE !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 🧭 Header
# ----------------------------------------
st.title("📈 Momentum Signal Dashboard")
st.markdown("Fetch **momentum trading signals** and visualize price vs momentum trends for multiple tickers.")

# ----------------------------------------
# 🧾 Input form
# ----------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    tickers = st.multiselect("Select Tickers", ALL_TICKERS, default=["TSLA", "AMZN", "F", "NVDA"])
with col2:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=90))
with col3:
    end_date = st.date_input("End Date", date.today())

col4, col5 = st.columns(2)
with col4:
    ma_type = st.selectbox("Moving Average Type", ["ema", "sma"], index=0)
with col5:
    method = st.selectbox("Signal Logic", ["standard", "ema2"], index=0)

# ----------------------------------------
# 🧩 Validation
# ----------------------------------------
if start_date >= end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()

if not tickers:
    st.error("⚠️ Please select at least one ticker.")
    st.stop()

# ----------------------------------------
# 🚀 Fetch Data from API
# ----------------------------------------
if st.button("Get Signals"):
    # Determine ES environment based on date range
    cutoff_date = date(2025, 12, 1)
    es_env = "staging" if end_date >= cutoff_date else "prod"
    
    # Split tickers into batches
    ticker_batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    num_batches = len(ticker_batches)
    
    st.info(f"Fetching {method.upper()} signals for **{len(tickers)} tickers** in {num_batches} batch(es)...")
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for batch_idx, batch_tickers in enumerate(ticker_batches, 1):
            status_text.text(f"Processing batch {batch_idx}/{num_batches} ({len(batch_tickers)} tickers)...")
            
            payload = {
                "query": {
                    "key": ENV_KEY,
                    "tickers": batch_tickers,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "ma_type": ma_type.lower(),
                    "method": method.lower(),
                },
                "es_env": es_env
            }
            
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                st.error(f"❌ Batch {batch_idx} failed: {response.status_code} - {response.text}")
                continue
            
            res_json = response.json()
            data = (
                res_json.get("data")
                or res_json.get("body", {}).get("data", [])
                or res_json.get("body")
            )
            
            if data:
                all_data.extend(data)
            
            progress_bar.progress(batch_idx / num_batches)
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            st.warning("⚠️ No signal data returned for this range.")
            st.stop()
        
        df_all = pd.DataFrame(all_data)
        if "date" in df_all.columns:
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        
        st.success(f"✅ Received {len(df_all)} total rows across {len(tickers)} tickers")

        # ----------------------------------------
        # 🧾 Per-Ticker Display
        # ----------------------------------------
        for ticker in tickers:
            st.subheader(f"📊 {ticker} Signals")
            df = df_all[df_all["ticker"] == ticker].copy()
            if df.empty:
                st.warning(f"No data found for {ticker}.")
                continue

            def color_signal(val):
                if isinstance(val, str):
                    if "STRONG BUY" in val:
                        return "background-color: #2ecc71; color: white"
                    if "BUY" in val:
                        return "background-color: #27ae60; color: white"
                    if "STRONG SELL" in val:
                        return "background-color: #e74c3c; color: white"
                    if "SELL" in val:
                        return "background-color: #c0392b; color: white"
                    if "SKIP" in val:
                        return "background-color: #7f8c8d; color: white"
                return "background-color: #95a5a6; color: white"

            st.dataframe(
                df.style.applymap(color_signal, subset=["Signal"]),
                use_container_width=True,
                height=400,
            )

            # Chart
            with st.expander(f"📉 {ticker} Price & Momentum Chart"):
                if "MomentumScore" in df.columns:
                    base = alt.Chart(df).encode(x="date:T")

                    momentum_line = (
                        base.mark_line(point=True, strokeWidth=2)
                        .encode(
                            y=alt.Y("MomentumScore:Q", axis=alt.Axis(title="Momentum Score")),
                            color=alt.Color(
                                "Signal:N",
                                scale=alt.Scale(
                                    domain=[
                                        "STRONG BUY",
                                        "BUY",
                                        "WATCH / HOLD",
                                        "SELL",
                                        "STRONG SELL",
                                    ],
                                    range=[
                                        "#2ecc71",
                                        "#27ae60",
                                        "#95a5a6",
                                        "#e74c3c",
                                        "#c0392b",
                                    ],
                                ),
                            ),
                            tooltip=["date:T", "Signal:N", "MomentumScore:Q", "Confidence:N"],
                        )
                    )

                    charts = [momentum_line]

                    if "price" in df.columns:
                        price_line = (
                            base.mark_line(color="#3498db", strokeDash=[3, 2])
                            .encode(
                                y=alt.Y("price:Q", axis=alt.Axis(title="Price (USD)", orient="right")),
                                tooltip=["date:T", "price:Q"],
                            )
                        )
                        charts.append(price_line)

                    final_chart = (
                        alt.layer(*charts)
                        .resolve_scale(y="independent")
                        .properties(
                            title=f"{ticker} Price vs Momentum ({method.upper()} | {ma_type.upper()})",
                            width=900,
                            height=400,
                        )
                    )
                    st.altair_chart(final_chart, use_container_width=True)
                else:
                    st.warning("No 'MomentumScore' column found for plotting.")

            # Individual CSV Download
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download {ticker} CSV",
                data=csv_data,
                file_name=f"{ticker}_{method}_{ma_type}_signals.csv",
                mime="text/csv",
            )
    
    except Exception as e:
        st.error(f"⚠️ Exception occurred: {e}")
