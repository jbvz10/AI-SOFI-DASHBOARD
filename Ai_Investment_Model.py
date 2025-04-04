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

@st.experimental_memo(ttl=120)
def trigger_autorefresh():
    return datetime.now()

trigger_autorefresh()

# Dashboard continues with market data and trading analytics