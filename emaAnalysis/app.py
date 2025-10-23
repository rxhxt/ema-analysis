import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta

# ----------------------------------------
# 🔧 Configuration
# ----------------------------------------
API_URL = "https://25becp9ef3.execute-api.ap-south-1.amazonaws.com/dev/ema-alerts"  # <-- replace with your actual endpoint

st.set_page_config(page_title="Momentum Signal Dashboard", layout="wide")

# ----------------------------------------
# 🧭 Header
# ----------------------------------------
st.title("📈 Momentum Signal Dashboard")
st.markdown("Enter parameters and fetch signals via your Lambda API endpoint.")

# ----------------------------------------
# 🧾 Input form
# ----------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", "TSLA")
with col2:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=90))
with col3:
    end_date = st.date_input("End Date", date.today())

env_key = st.selectbox("Environment Key", ["prod", "dev", "stable"], index=0)

if start_date >= end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()

# ----------------------------------------
# 🚀 Fetch from API
# ----------------------------------------
if st.button("Get Signals"):
    payload = {
        "query": {
            "key": env_key,
            "ticker": ticker.upper(),
            "start_date": str(start_date),
            "end_date": str(end_date)
        }
    }

    with st.spinner("Fetching data from Lambda API..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                st.error(f"❌ Error {response.status_code}: {response.text}")
            else:
                data = response.json().get("data", [])
                if not data:
                    st.warning("No signal data returned for this range.")
                else:
                    df = pd.DataFrame(data)
                    # display results
                    st.success(f"✅ Received {len(df)} rows for {ticker.upper()}")
                    
                    # format table
                    def color_signal(val):
                        if "STRONG BUY" in val: return "background-color: #2ecc71; color: white"
                        if "BUY" in val: return "background-color: #27ae60; color: white"
                        if "STRONG SELL" in val: return "background-color: #e74c3c; color: white"
                        if "SELL" in val: return "background-color: #c0392b; color: white"
                        if "SKIP" in val: return "background-color: #7f8c8d; color: white"
                        return "background-color: #95a5a6; color: white"

                        # Show dataframe
                    st.dataframe(
                        df.style.applymap(color_signal, subset=["Signal"]),
                        use_container_width=True,
                        height=600
                    )

                    # Optional chart
                    with st.expander("📉 Show Momentum Chart"):
                        import altair as alt
                        chart = alt.Chart(df).mark_line(point=True).encode(
                            x="date:T",
                            y="MomentumScore:Q",
                            color=alt.Color("Signal:N",
                                            scale=alt.Scale(
                                                domain=["STRONG BUY", "BUY", "WATCH / HOLD", "SELL", "STRONG SELL"],
                                                range=["#2ecc71", "#27ae60", "#95a5a6", "#e74c3c", "#c0392b"]
                                            ))
                        ).properties(width=900, height=400)
                        st.altair_chart(chart, use_container_width=True)

                    # CSV download
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", data=csv, file_name=f"{ticker}_signals.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Exception: {e}")
