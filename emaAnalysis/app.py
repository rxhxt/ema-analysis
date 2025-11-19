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

st.set_page_config(page_title="Momentum Signal Dashboard", layout="wide")

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
    tickers_input = st.text_input("Tickers (comma-separated)", "TSLA, AMZN, F, NVDA").upper()
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

# Parse ticker list
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.error("⚠️ Please enter at least one valid ticker symbol.")
    st.stop()

# ----------------------------------------
# 🚀 Fetch Data from API
# ----------------------------------------
if st.button("Get Signals"):
    payload = {
        "query": {
            "key": ENV_KEY,
            "tickers": tickers,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "ma_type": ma_type.lower(),
            "method": method.lower(),
        }
    }

    st.info(f"Fetching {method.upper()} signals for **{', '.join(tickers)}** ...")
    with st.spinner("Processing... please wait"):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                st.error(f"❌ Error {response.status_code}: {response.text}")
                st.stop()

            res_json = response.json()
            data = (
                res_json.get("data")
                or res_json.get("body", {}).get("data", [])
                or res_json.get("body")
            )

            if not data:
                st.warning("⚠️ No signal data returned for this range.")
                st.stop()

            df_all = pd.DataFrame(data)
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
