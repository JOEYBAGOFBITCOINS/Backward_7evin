import warnings
warnings.filterwarnings("ignore")

# ===== Core Imports =====
import time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# Optional forecasting libs
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATS_OK = True
except Exception:
    STATS_OK = False

# ===== Constants =====
APP_TITLE = "Backward 7evin"
ASSETS = {"BTC-USD": "Bitcoin", "GC=F": "Gold", "DX-Y.NYB": "USD"}

# ===== Page Theme =====
st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp{
  background: radial-gradient(circle at 10% 20%, #0f0f14 0%, #0a0a0e 100%);
  color:#e6e6e6;
}
.glass{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.45);
  backdrop-filter: blur(10px);
  padding: 18px 20px;
}
.signal{
  font-weight:700;font-size:22px;letter-spacing:.3px;
  padding:8px 14px;border-radius:12px;display:inline-block;
}
.long{background:rgba(22,163,74,0.2);border:1px solid rgba(22,163,74,0.5);}
.short{background:rgba(220,38,38,0.2);border:1px solid rgba(220,38,38,0.5);}
.hold{background:rgba(148,163,184,0.2);border:1px solid rgba(148,163,184,0.5);}
.metric{font-size:14px;color:#b3b3b3;}
.bar-wrap{height:10px;border-radius:8px;background:#222;overflow:hidden}
.bar{height:10px;background:#3b82f6}
</style>
""", unsafe_allow_html=True)

# ===== Title Row with Custom Gold Logo =====
col1, col2 = st.columns([8, 2], vertical_alignment="center")

with col1:
    st.markdown("""
    <h1 style='font-size:52px;font-weight:900;
    color:#BF9B61; /* custom gold tone */
    background:transparent;
    display:inline-block;padding:10px 20px;border-radius:8px;'>
    Backward <span style='display:inline-block;transform:scaleX(-1);'>7</span>evin
    </h1>
    """, unsafe_allow_html=True)

with col2:
    st.image("/workspaces/Backward_7evin/backward7evin_Logo.png", use_container_width=True)

st.markdown("""
<style>
[data-testid="stImage"] img {
    max-height: 140px;
    object-fit: contain;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===== Auto Refresh =====
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="live_refresh")
except Exception:
    now = time.time()
    if "last_refresh_ts" not in st.session_state:
        st.session_state["last_refresh_ts"] = now
    elif now - st.session_state["last_refresh_ts"] > 60:
        st.session_state["last_refresh_ts"] = now
        st.rerun()

# ===== Helper Functions =====
def fmt_money(x: float) -> str:
    try:
        if abs(x) >= 1_000_000:
            return f"${x/1_000_000:.2f}M"
        elif abs(x) >= 1_000:
            return f"${x:,.0f}"
        else:
            return f"${x:,.2f}"
    except Exception:
        return "-"

@st.cache_data(ttl=900)
def fetch_prices(period="180d", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(list(ASSETS.keys()), period=period, interval=interval, progress=False)["Close"]
        df.columns = [ASSETS.get(c, c) for c in df.columns]
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def compute_rsi(series: pd.Series, window: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(df):
    feats = pd.DataFrame(index=df.index)
    for col in df.columns:
        feats[f"{col}_ret1"] = df[col].pct_change()
        feats[f"{col}_ret5"] = df[col].pct_change(5)
        feats[f"{col}_vol10"] = df[col].pct_change().rolling(10).std()
    if "Bitcoin" in df:
        feats["BTC_RSI"] = compute_rsi(df["Bitcoin"])
        feats["BTC_MA7"] = df["Bitcoin"].rolling(7).mean()
        feats["BTC_MA21"] = df["Bitcoin"].rolling(21).mean()
        feats["BTC_MA_diff"] = feats["BTC_MA7"] - feats["BTC_MA21"]
    return feats.dropna()

def label_target(df):
    return (df["Bitcoin"].shift(-1) > df["Bitcoin"]).astype(int)

def train_rf(df):
    X = build_features(df)
    y = label_target(df).reindex(X.index)
    data = pd.concat([X, y.rename("target")], axis=1).dropna()
    if len(data) < 30:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target"]), data["target"], test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train_s, y_train)
    latest = scaler.transform(X.tail(1))
    proba = rf.predict_proba(latest)[0]
    pred = rf.predict(latest)[0]
    return {"signal": "LONG" if pred == 1 else "SHORT", "confidence": float(max(proba))*100}

def train_ensemble(df):
    X = build_features(df)
    y = label_target(df).reindex(X.index)
    data = pd.concat([X, y.rename("target")], axis=1).dropna()
    if len(data) < 30:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target"]), data["target"], test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=1000)
    ens = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting="soft")
    ens.fit(X_train_s, y_train)
    latest = scaler.transform(X.tail(1))
    proba = ens.predict_proba(latest)[0]
    pred = ens.predict(latest)[0]
    votes = {
        "Random Forest": "LONG" if rf.fit(X_train_s, y_train).predict(latest)[0] == 1 else "SHORT",
        "Gradient Boost": "LONG" if gb.fit(X_train_s, y_train).predict(latest)[0] == 1 else "SHORT",
        "Logistic Reg": "LONG" if lr.fit(X_train_s, y_train).predict(latest)[0] == 1 else "SHORT",
    }
    return {"signal": "LONG" if pred == 1 else "SHORT",
            "confidence": float(max(proba))*100,
            "votes": votes}

# ===== Sidebar =====
with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History Window", ["90d", "180d", "1y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)
    st.divider()
    st.subheader("Models")
    use_arima = st.checkbox("ARIMA Forecast", True)
    use_hw = st.checkbox("Holt–Winters Forecast", True)
    use_rf = st.checkbox("Random Forest Signal", True)
    use_ens = st.checkbox("Use Ensemble Model", True)
    st.divider()
    st.caption("Live data refresh is set to 1 minute.")

# ===== Load & Train =====
with st.spinner("Fetching live market data..."):
    raw = fetch_prices(period=period, interval=interval)

if raw.empty:
    st.error("No data available. Try another window or interval.")
    st.stop()

rf_res = train_rf(raw) if use_rf else None
ens_res = train_ensemble(raw) if use_ens else None

# ===== Top Summary =====
st.markdown("<div class='glass'>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
latest = raw.iloc[-1]
c1.metric("Bitcoin", fmt_money(latest['Bitcoin']))
c2.metric("Gold", fmt_money(latest['Gold']))
c3.metric("USD Index", f"{latest['USD']:.2f}")

def action_from_signal(sig, conf):
    if conf < 60:
        return "⚪ HOLD — Mixed conditions"
    if sig == "LONG":
        return f"🟢 LONG ({conf:.1f}%)"
    return f"🔴 SHORT ({conf:.1f}%)"

if ens_res:
    action_text = action_from_signal(ens_res["signal"], ens_res["confidence"])
elif rf_res:
    action_text = action_from_signal(rf_res["signal"], rf_res["confidence"])
else:
    action_text = "⚪ HOLD — Mixed conditions"

c4.markdown(f"**AI Direction:** {action_text}")
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# ===== Tabs =====
tabs = st.tabs(["Charts", "Forecasts", "Ensemble"])

# Charts
with tabs[0]:
    st.subheader("Price Charts")
    for ticker, name in ASSETS.items():
        st.markdown(f"### {name} Price History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw.index, y=raw[name], mode="lines", name=name))
        fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# Forecasts
with tabs[1]:
    st.subheader("Forecasts — next 5 steps")

    for ticker, name in ASSETS.items():
        series = raw[name]
        a, h = pd.Series(dtype=float), pd.Series(dtype=float)
        if use_arima and STATS_OK:
            try: a = ARIMA(series.dropna(), order=(1,1,1)).fit().forecast(5)
            except Exception: pass
        if use_hw and STATS_OK:
            try: h = ExponentialSmoothing(series.dropna(), trend="add").fit().forecast(5)
            except Exception: pass

        if not a.empty or not h.empty:
            a_change = (a.iloc[-1] - series.iloc[-1]) / series.iloc[-1] * 100 if not a.empty else 0
            h_change = (h.iloc[-1] - series.iloc[-1]) / series.iloc[-1] * 100 if not h.empty else 0
            avg_change = np.mean([a_change, h_change])
            if avg_change > 0.1:
                icon, desc = "📈", "likely to rise"
            elif avg_change < -0.1:
                icon, desc = "📉", "may decline"
            else:
                icon, desc = "⚖️", "flat trend"
            st.write(f"{icon} **{name}** — {desc} (ARIMA: {a_change:+.2f}%, HW: {h_change:+.2f}%)")
        else:
            st.write(f"{name}: insufficient data to forecast.")

# Ensemble
with tabs[2]:
    st.subheader("Ensemble Results")
    if ens_res:
        sig, conf = ens_res["signal"], ens_res["confidence"]
        css = "long" if sig == "LONG" else "short" if conf >= 60 else "hold"
        st.markdown(f"<div class='signal {css}'>Signal: {sig if conf>=60 else 'HOLD'} ({conf:.1f}%)</div>", unsafe_allow_html=True)
        st.write("**Votes:**")
        for model, vote in ens_res["votes"].items():
            icon = "🟢" if vote == "LONG" else "🔴"
            st.write(f"• {model}: {icon} {vote}")
        st.markdown("<div class='bar-wrap'><div class='bar' style='width:%s%%;'></div></div>" % f"{min(max(conf,0),100):.0f}", unsafe_allow_html=True)
        st.caption("Confidence reflects model agreement. (<60% = HOLD)")
    else:
        st.info("Not enough data or model disabled. Enable 'Use Ensemble Model' to see combined signal.")
