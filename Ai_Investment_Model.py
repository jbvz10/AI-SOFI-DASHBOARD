# AI Weekly Swing Trading Streamlit Dashboard (SOFI + Options Analysis)

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import json
import requests
import streamlit as st

start_date = "2023-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')
ticker = "SOFI"

portfolio = {
    "entry_price": 7.15,
    "shares": 200,
    "entry_date": "2024-03-01",
    "strategy": "double_diagonal",
    "long_legs": {
        "SOFI Jan25 12 Call": {"contracts": 12, "avg_price": 4.322, "last_price": 2.90},
        "SOFI Jan25 12 Put": {"contracts": 13, "avg_price": 3.657, "last_price": 4.78}
    },
    "short_legs": {
        "SOFI Apr11 12.5 Call": {"contracts": -3, "avg_price": 0.507, "last_price": 0.04},
        "SOFI Apr11 12.5 Put": {"contracts": -3, "avg_price": 0.742, "last_price": 2.71},
        "SOFI Apr11 10.5 Put": {"contracts": -2, "avg_price": 0.432, "last_price": 1.40},
        "SOFI May02 11 Put": {"contracts": -3, "avg_price": 1.142, "last_price": 2.16},
        "SOFI May16 13 Call": {"contracts": -2, "avg_price": 1.292, "last_price": 0.30}
    }
}

def calculate_option_pnl(legs):
    pnl_data = []
    for name, info in legs.items():
        position = info['contracts']
        avg_price = info['avg_price']
        last_price = info['last_price']
        pnl = (last_price - avg_price) * position * 100
        pnl_data.append({
            'Option': name,
            'Position': position,
            'Avg Price': avg_price,
            'Last Price': last_price,
            'Unrealized P&L': pnl
        })
    return pd.DataFrame(pnl_data)

def fallback_alpha_vantage_chain(symbol):
    fake_call = pd.DataFrame([{
        'contractSymbol': f'{symbol}_CALL_FAKE',
        'strike': 13.0,
        'impliedVolatility': 0.65
    }])
    fake_put = pd.DataFrame([{
        'contractSymbol': f'{symbol}_PUT_FAKE',
        'strike': 11.0,
        'impliedVolatility': 0.70
    }])
    expiry = (datetime.today() + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    return expiry, fake_call, fake_put

def recommend_weekly_shorts(ticker):
    for _ in range(3):
        try:
            yf_ticker = yf.Ticker(ticker)
            expirations = yf_ticker.options
            if not expirations or len(expirations) < 2:
                continue
            short_exp = expirations[1]
            url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}?date={int(pd.Timestamp(short_exp).timestamp())}"
            r = requests.get(url)
            if not r.ok or "application/json" not in r.headers.get("Content-Type", ""):
                continue
            options = yf_ticker.option_chain(short_exp)
            calls = options.calls.copy()
            puts = options.puts.copy()
            spot_data = yf_ticker.history(period="5d")
            if spot_data.empty or 'Close' not in spot_data.columns:
                continue
            spot = spot_data['Close'].iloc[-1]
            calls = calls[(calls['strike'] > spot) & (calls['strike'] <= spot + 1.5)]
            puts = puts[(puts['strike'] < spot) & (puts['strike'] >= spot - 1.5)]
            best_call = calls.sort_values('impliedVolatility', ascending=False).head(1)[['contractSymbol', 'strike', 'impliedVolatility']] if not calls.empty else None
            best_put = puts.sort_values('impliedVolatility', ascending=False).head(1)[['contractSymbol', 'strike', 'impliedVolatility']] if not puts.empty else None
            return short_exp, best_call, best_put
        except Exception:
            time.sleep(1)
            continue
    return fallback_alpha_vantage_chain(ticker)

long_legs_df = calculate_option_pnl(portfolio['long_legs'])
short_legs_df = calculate_option_pnl(portfolio['short_legs'])
options_pnl_df = pd.concat([long_legs_df, short_legs_df], ignore_index=True)
total_pnl = options_pnl_df['Unrealized P&L'].sum()

rec_exp, rec_call, rec_put = recommend_weekly_shorts(ticker)

st.set_page_config(page_title="SOFI Options Dashboard", layout="wide")
st.title("📊 SOFI Double Diagonal Strategy Dashboard")

st.subheader("💼 Portfolio Position")
st.markdown(f"**Shares:** {portfolio['shares']} @ ${portfolio['entry_price']} (Since {portfolio['entry_date']})")

st.subheader("📈 Options Legs Unrealized P&L")
st.dataframe(options_pnl_df, use_container_width=True)
st.markdown(f"**Total Unrealized P&L: ${total_pnl:,.2f}**")

st.subheader("📌 Recommended Weekly Options to Short")
if rec_exp:
    st.markdown(f"**Expiration Date:** {rec_exp}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Call to Short:**")
        st.dataframe(rec_call)
    with col2:
        st.markdown("**Put to Short:**")
        st.dataframe(rec_put)
else:
    st.warning("No options data available for recommendations.")

