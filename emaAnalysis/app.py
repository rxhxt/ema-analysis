import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
import altair as alt

# ----------------------------------------
# 🔧 Configuration
# ----------------------------------------
API_URL = "https://25becp9ef3.execute-api.ap-south-1.amazonaws.com/dev/ema-alerts"

st.set_page_config(page_title="Momentum Signal Dashboard", layout="wide")

# ----------------------------------------
# 🧭 Header
# ----------------------------------------
st.title("📈 Momentum Signal Dashboard")
st.markdown("Use this dashboard to fetch **momentum trading signals** and visualize price vs momentum trends.")

# ----------------------------------------
# 🧾 Input form
# ----------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", "TSLA").upper()
with col2:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=90))
with col3:
    end_date = st.date_input("End Date", date.today())

col4, col5, col6 = st.columns(3)
with col4:
    env_key = st.selectbox("Environment Key", ["prod", "dev", "stable"], index=0)
with col5:
    ma_type = st.selectbox("Moving Average Type", ["ema", "sma"], index=0)
with col6:
    method = st.selectbox("Signal Logic", ["standard", "ema2"], index=0)

# ----------------------------------------
# 🧩 Validation
# ----------------------------------------
if start_date >= end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()

# ----------------------------------------
# 🚀 Fetch Data from API
# ----------------------------------------
if st.button("Get Signals"):
    payload = {
        "query": {
            "key": env_key,
            "ticker": ticker,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "ma_type": ma_type.lower(),
            "method": method.lower()
        }
    }

    st.info(f"Fetching {method.upper()} signals for **{ticker}** ({ma_type.upper()}) ...")
    with st.spinner("Processing... please wait"):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                st.error(f"❌ Error {response.status_code}: {response.text}")
            else:
                res_json = response.json()
                data = (
                    res_json.get("data")
                    or res_json.get("body", {}).get("data", [])
                    or res_json.get("body")
                )

                if not data:
                    st.warning("⚠️ No signal data returned for this range.")
                    st.stop()

                # Convert to DataFrame
                df = pd.DataFrame(data)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

                st.success(f"✅ Received {len(df)} rows for {ticker} ({method.upper()} | {ma_type.upper()})")

                # ----------------------------------------
                # 🎨 Data styling
                # ----------------------------------------
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

                st.subheader("📊 Signal Table")
                st.dataframe(
                    df.style.applymap(color_signal, subset=["Signal"]),
                    use_container_width=True,
                    height=600,
                )

                # ----------------------------------------
                # 📈 Combined Chart: Price + Momentum
                # ----------------------------------------
                with st.expander("📉 Show Price & Momentum Chart"):
                    if "MomentumScore" in df.columns:
                        base = alt.Chart(df).encode(x="date:T")

                        # Momentum line (colored by signal)
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
                                tooltip=[
                                    "date:T",
                                    "Signal:N",
                                    "MomentumScore:Q",
                                    "Confidence:N",
                                ],
                            )
                        )

                        charts = [momentum_line]

                        # ✅ Add price line if available
                        if "price" in df.columns:
                            price_line = (
                                base.mark_line(color="#3498db", strokeDash=[3, 2])
                                .encode(
                                    y=alt.Y("price:Q", axis=alt.Axis(title="Price (USD)", orient="right")),
                                    tooltip=["date:T", "price:Q"]
                                )
                            )
                            charts.append(price_line)

                        final_chart = alt.layer(*charts).resolve_scale(y="independent").properties(
                            title=f"{ticker} Price vs Momentum ({method.upper()} | {ma_type.upper()})",
                            width=900,
                            height=400,
                        )

                        st.altair_chart(final_chart, use_container_width=True)
                    else:
                        st.warning("No 'MomentumScore' column found for plotting.")

                # ----------------------------------------
                # 💾 CSV Download
                # ----------------------------------------
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"{ticker}_{method}_{ma_type}_signals.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"⚠️ Exception occurred: {e}")
