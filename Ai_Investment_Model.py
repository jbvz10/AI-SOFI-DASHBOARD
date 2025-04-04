# AI Weekly Swing Trading Streamlit Dashboard (SOFI + Options Analysis)

import streamlit as st
st.set_page_config(page_title="SOFI Options Dashboard", layout="wide")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import json
import requests

start_date = "2023-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

@st.cache_data(ttl=120)
def trigger_autorefresh():
    return datetime.now()

trigger_autorefresh()

ticker = st.sidebar.text_input("Ticker Symbol", value="SOFI").upper()
refresh = st.sidebar.button("🔁 Refresh Dashboard")

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

def fetch_earnings_calendar():
    calendar = [
        {"ticker": "AAPL", "date": "2024-04-25"},
        {"ticker": "MSFT", "date": "2024-04-23"},
        {"ticker": "TSLA", "date": "2024-04-24"},
    ]
    return pd.DataFrame(calendar)

def scan_top_volatility_ideas():
    tickers = ["AAPL", "TSLA", "NVDA", "AMD", "META"]
    scores = []
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="7d")
            if data.empty:
                continue
            returns = data['Close'].pct_change().dropna()
            vol = returns.std()
            score = vol * 100
            scores.append({"Ticker": t, "Volatility Score": round(score, 2)})
        except:
            continue
    return pd.DataFrame(scores).sort_values("Volatility Score", ascending=False).head(3)

if refresh:
    st.experimental_rerun()

long_legs_df = calculate_option_pnl(portfolio['long_legs'])
short_legs_df = calculate_option_pnl(portfolio['short_legs'])
options_pnl_df = pd.concat([long_legs_df, short_legs_df], ignore_index=True)
total_pnl = options_pnl_df['Unrealized P&L'].sum()
long_total = long_legs_df['Unrealized P&L'].sum()
short_total = short_legs_df['Unrealized P&L'].sum()

rec_exp, rec_call, rec_put = recommend_weekly_shorts(ticker)

st.title("📊 SOFI Double Diagonal Strategy Dashboard")

# Market Index Summary
index_data = {}
sparkline_data = {}
for symbol, label in zip(["^DJI", "^GSPC", "^IXIC"], ["Dow Jones", "S&P 500", "NASDAQ"]):
    idx_hist = yf.Ticker(symbol).history(period="5d")['Close']
    if len(idx_hist) >= 2:
        latest = idx_hist.iloc[-1]
        prev = idx_hist.iloc[-2]
        change = latest - prev
        pct_change = (change / prev) * 100
        index_data[label] = (latest, change, pct_change)
        sparkline_data[label] = idx_hist
for symbol, label in zip(["^DJI", "^GSPC", "^IXIC"], ["Dow Jones", "S&P 500", "NASDAQ"]):
    idx_hist = yf.Ticker(symbol).history(period="2d")['Close']
    if len(idx_hist) >= 2:
        latest = idx_hist.iloc[-1]
        prev = idx_hist.iloc[-2]
        change = latest - prev
        pct_change = (change / prev) * 100
        index_data[label] = (latest, change, pct_change)
index_cols = st.columns(3)
for i, (label, (value, change, pct)) in enumerate(index_data.items()):
    delta = f"{change:+.2f} ({pct:+.2f}%)"
    color = "normal"
    with index_cols[i]:
        st.metric(label, f"{value:,.2f}", delta, delta_color=color)
        st.line_chart(sparkline_data[label], use_container_width=True)

st.subheader("💼 Portfolio Position")
st.markdown(f"**Shares:** {portfolio['shares']} @ ${portfolio['entry_price']} (Since {portfolio['entry_date']})")

st.subheader("📉 SOFI Price Chart")
price_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
st.line_chart(price_data['Close'], use_container_width=True)

st.subheader("📈 Options Legs Unrealized P&L")
st.dataframe(options_pnl_df, use_container_width=True)
st.markdown(f"**Total Unrealized P&L: ${total_pnl:,.2f}**")
st.markdown(f"🔹 Long Legs P&L: ${long_total:,.2f}  |  🔸 Short Legs P&L: ${short_total:,.2f}")

csv = options_pnl_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download P&L Data", csv, "sofi_pnl.csv", "text/csv")

st.subheader("🧠 Sentiment Score")
analyzer = SentimentIntensityAnalyzer()
headline = "SOFI stock rises after earnings beat expectations"
sentiment = analyzer.polarity_scores(headline)
st.metric("Headline Sentiment", sentiment['compound'])

st.subheader("📌 Recommended Weekly Options to Short")
if rec_exp is not None:
    st.markdown(f"**Expiration Date:** {rec_exp}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Call to Short:**")
        if rec_call is not None:
            st.dataframe(rec_call)
        else:
            st.info("No suitable call option found.")
    with col2:
        st.markdown("**Put to Short:**")
        if rec_put is not None:
            st.dataframe(rec_put)
        else:
            st.info("No suitable put option found.")
else:
    st.warning("No options data available for recommendations.")

st.subheader("📊 Trade Performance Tracker")
if 'performance_log.csv' not in st.session_state:
    st.session_state['performance_log.csv'] = pd.DataFrame(columns=['Date', 'Trade', 'PnL'])

with st.expander("📝 Log New Trade"):
    trade_text = st.text_input("Trade Description")
    trade_pnl = st.number_input("Realized P&L", step=0.01)
    if st.button("Log Trade"):
        new_log = pd.DataFrame([[datetime.now().strftime('%Y-%m-%d'), trade_text, trade_pnl]], columns=['Date', 'Trade', 'PnL'])
        st.session_state['performance_log.csv'] = pd.concat([st.session_state['performance_log.csv'], new_log], ignore_index=True)
        st.success("Trade logged!")

filter_df = st.session_state['performance_log.csv']
with st.expander("🔍 Filter Trade Log"):
    selected_trade = st.multiselect("Filter by Trade", options=filter_df['Trade'].unique())
    min_date = st.date_input("Start Date", value=datetime.today())
    max_date = st.date_input("End Date", value=datetime.today())
    min_pnl = st.number_input("Min P&L", value=float(filter_df['PnL'].min() if not filter_df.empty else 0))
    max_pnl = st.number_input("Max P&L", value=float(filter_df['PnL'].max() if not filter_df.empty else 0))
    if selected_trade:
        filter_df = filter_df[filter_df['Trade'].isin(selected_trade)]
    filter_df['Date'] = pd.to_datetime(filter_df['Date'])
    filter_df = filter_df[(filter_df['Date'] >= pd.to_datetime(min_date)) & (filter_df['Date'] <= pd.to_datetime(max_date))]
    filter_df = filter_df[(filter_df['PnL'] >= min_pnl) & (filter_df['PnL'] <= max_pnl)]
st.dataframe(filter_df, use_container_width=True)

st.download_button("⬇️ Download Trade History", filter_df.to_csv(index=False).encode('utf-8'), "trade_log.csv")

st.subheader("📈 Weekly Performance Summary")
if not filter_df.empty:
    weekly_summary = filter_df.copy()
    weekly_summary['Week'] = weekly_summary['Date'].dt.to_period('W').astype(str)
    agg = weekly_summary.groupby('Week')['PnL'].agg(['count', 'sum']).reset_index()
    agg.rename(columns={'count': 'Trades', 'sum': 'Total P&L'}, inplace=True)
    st.dataframe(agg, use_container_width=True)

st.subheader("📡 AI Alerts & Commentary")
commentary = []
if sentiment['compound'] > 0.4:
    commentary.append("🔔 Positive news sentiment detected — potential for short-term upside.")
if rsi is not None:
    if rsi < 30:
        commentary.append("📈 RSI indicates oversold — bounce expected.")
    elif rsi > 70:
        commentary.append("⚠️ RSI is overbought — caution warranted.")
if not commentary:
    commentary.append("No strong signals detected.")
for line in commentary:
    st.write(line)

st.subheader("💰 Smart Trade Ideas This Week")
rsi_data = price_data['Close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.pct_change().mean() / x.pct_change().std()))))
rsi = rsi_data.iloc[-1] if not rsi_data.empty else None

if sentiment['compound'] > 0.3 and price_data['Close'].iloc[-1] > price_data['Close'].mean():
    st.success("📈 Bullish sentiment + price momentum. Consider selling a put credit spread below support.")
elif sentiment['compound'] < -0.3 and price_data['Close'].iloc[-1] < price_data['Close'].mean():
    st.warning("📉 Bearish bias. Consider a call credit spread or bearish diagonal.")
else:
    st.info("➖ Neutral zone. If IV is high, consider neutral strategies like strangles or calendars.")

if rsi is not None:
    st.markdown(f"**Current RSI (14): {rsi:.2f}**")
    if rsi < 30:
        st.success("📊 RSI is oversold (<30). Watch for a potential bullish reversal.")
    elif rsi > 70:
        st.warning("⚠️ RSI is overbought (>70). Watch for potential pullback or fade.")
    else:
        st.info("RSI in neutral range.")

st.subheader("📅 Earnings Calendar (Key Stocks)")
st.dataframe(fetch_earnings_calendar(), use_container_width=True)

st.subheader("🔥 Top Weekly Volatility Trading Ideas")
st.dataframe(scan_top_volatility_ideas(), use_container_width=True)

st.subheader("📃 Full Options Chain Viewer")
yf_ticker = yf.Ticker(ticker)
try:
    available_expiries = yf_ticker.options
    selected_expiry = st.selectbox("Select Expiration Date", options=available_expiries)
    if selected_expiry:
        opt_chain = yf_ticker.option_chain(selected_expiry)
        calls, puts = opt_chain.calls.copy(), opt_chain.puts.copy()
        spot_price = yf_ticker.history(period='1d')['Close'].iloc[-1]

        st.markdown("### 🎯 Filters")
        range_pct = st.slider("Strike Range (% from Spot Price)", 0, 100, 15)
        min_oi = st.number_input("Minimum Open Interest", min_value=0, value=100)

        calls = calls[(calls['strike'] >= spot_price * (1 - range_pct / 100)) & 
                      (calls['strike'] <= spot_price * (1 + range_pct / 100))]
        calls = calls[calls['openInterest'] >= min_oi]

        puts = puts[(puts['strike'] >= spot_price * (1 - range_pct / 100)) & 
                    (puts['strike'] <= spot_price * (1 + range_pct / 100))]
        puts = puts[puts['openInterest'] >= min_oi]

        def highlight_oi(val):
            return 'background-color: #ffe599' if val > min_oi * 2 else ''

        st.markdown("**📈 Filtered Calls**")
        st.dataframe(calls[['contractSymbol', 'strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest']]
                     .style.applymap(highlight_oi, subset=['openInterest']), use_container_width=True)

        st.markdown("**📉 Filtered Puts**")
        st.dataframe(puts[['contractSymbol', 'strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest']]
                     .style.applymap(highlight_oi, subset=['openInterest']), use_container_width=True)
except Exception as e:
    st.error(f"Error loading options chain: {e}")
