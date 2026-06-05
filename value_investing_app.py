"""
Value Investing Toolkit — Streamlit App
Based on: Security Analysis and Business Valuation on Wall Street (Hooke)
Data: Alpha Vantage API (free tier)
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Value Investing Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
  .signal-buy  { color: #00e676; font-weight: 600; }
  .signal-sell { color: #ff5252; font-weight: 600; }
  .signal-hold { color: #ffd740; font-weight: 600; }
  .signal-na   { color: #888; }
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; color: #888;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #2d2d2d;
    padding-bottom: 6px; margin: 20px 0 12px 0;
  }
  .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_currency(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v/1e12:.{decimals}f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.{decimals}f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.{decimals}f}M"
    return f"${v:,.{decimals}f}"

def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:.{decimals}f}%"

def fmt_x(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}×"

def safe_div(a, b):
    try:
        if b is None or b == 0 or np.isnan(float(b)):
            return None
        return float(a) / float(b)
    except Exception:
        return None

def safe_float(v, default=None):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except Exception:
        return default

def signal_html(label, cls):
    return f'<span class="{cls}">{label}</span>'


# ── Alpha Vantage fetch functions ─────────────────────────────────────────────

BASE = "https://www.alphavantage.co/query"

@st.cache_data(ttl=600, show_spinner=False)
def av_get(function, ticker, api_key, **kwargs):
    params = {"function": function, "symbol": ticker, "apikey": api_key}
    params.update(kwargs)
    r = requests.get(BASE, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "Note" in data:
        raise RuntimeError("Alpha Vantage rate limit hit — wait 1 minute and try again.")
    if "Information" in data:
        raise RuntimeError("Alpha Vantage: " + data["Information"])
    if "Error Message" in data:
        raise RuntimeError(f"Ticker '{ticker}' not found. Check the symbol.")
    return data

@st.cache_data(ttl=600, show_spinner=False)
def fetch_all(ticker, api_key):
    import time
    results = []
    calls = [
        ("OVERVIEW",          {}),
        ("INCOME_STATEMENT",  {}),
        ("BALANCE_SHEET",     {}),
        ("CASH_FLOW",         {}),
        ("GLOBAL_QUOTE",      {}),
        ("TIME_SERIES_DAILY", {"outputsize": "compact"}),
    ]
    for i, (function, kwargs) in enumerate(calls):
        if i > 0:
            time.sleep(1.2)   # stay under 1 req/sec free tier limit
        results.append(av_get(function, ticker, api_key, **kwargs))
    return tuple(results)


# ── Extract financials from AV responses ──────────────────────────────────────

def latest(report_list):
    """Return the most recent annual report dict from AV annualReports list."""
    try:
        return report_list[0] if report_list else {}
    except Exception:
        return {}

def extract_financials(overview, income, balance, cashflow, quote, daily):
    d = {}

    q  = quote.get("Global Quote", {})
    inc = latest(income.get("annualReports", []))
    bal = latest(balance.get("annualReports", []))
    cf  = latest(cashflow.get("annualReports", []))

    # ── Price & market ──
    d["price"]       = safe_float(q.get("05. price"))
    d["shares"]      = safe_float(overview.get("SharesOutstanding"))
    d["market_cap"]  = safe_float(overview.get("MarketCapitalization"))
    d["beta"]        = safe_float(overview.get("Beta"))
    d["sector"]      = overview.get("Sector", "N/A")
    d["industry"]    = overview.get("Industry", "N/A")
    d["name"]        = overview.get("Name", overview.get("Symbol", ""))

    # ── P&L ──
    d["revenue"]     = safe_float(inc.get("totalRevenue"))
    d["net_income"]  = safe_float(inc.get("netIncome"))
    d["ebitda"]      = safe_float(overview.get("EBITDA"))
    d["eps"]         = safe_float(overview.get("EPS"))

    # ── Balance sheet ──
    d["total_debt"]  = safe_float(bal.get("shortLongTermDebtTotal")) or \
                       (safe_float(bal.get("longTermDebt"), 0) or 0) + \
                       (safe_float(bal.get("shortTermDebt"), 0) or 0)
    d["cash"]        = safe_float(bal.get("cashAndCashEquivalentsAtCarryingValue")) or \
                       safe_float(bal.get("cashAndShortTermInvestments"), 0)
    d["total_equity"]= safe_float(bal.get("totalShareholderEquity"))
    d["book_value"]  = safe_div(d["total_equity"], d["shares"]) if d["total_equity"] and d["shares"] else \
                       safe_float(overview.get("BookValue"))

    # ── Cash flow ──
    d["capex"]       = abs(safe_float(cf.get("capitalExpenditures"), 0) or 0)
    d["operating_cf"]= safe_float(cf.get("operatingCashflow"), 0)
    d["dividends_paid"] = abs(safe_float(cf.get("dividendPayout"), 0) or 0)

    # Dividend per share approximation
    d["dividend_ttm"] = safe_div(d["dividends_paid"], d["shares"]) if d["shares"] else \
                        safe_float(overview.get("DividendPerShare"), 0)

    # ── Multiples from overview ──
    d["pe_ratio"]    = safe_float(overview.get("PERatio"))
    d["pb_ratio"]    = safe_float(overview.get("PriceToBookRatio"))
    d["ps_ratio"]    = safe_float(overview.get("PriceToSalesRatioTTM"))
    d["ev"]          = safe_float(overview.get("EVToEBITDA"))   # ratio; we'll compute $ EV below
    d["ev_dollars"]  = (d["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0) \
                       if d["market_cap"] else None
    d["dividend_yield"]  = safe_float(overview.get("DividendYield"), 0)

    # ── Growth ──
    d["revenue_growth"]  = safe_float(overview.get("QuarterlyRevenueGrowthYOY"))
    d["earnings_growth"] = safe_float(overview.get("QuarterlyEarningsGrowthYOY"))

    # ── Profitability ──
    d["roe"]             = safe_float(overview.get("ReturnOnEquityTTM"))
    d["roa"]             = safe_float(overview.get("ReturnOnAssetsTTM"))
    d["gross_margin"]    = safe_float(overview.get("GrossProfitTTM")) and \
                           safe_div(safe_float(overview.get("GrossProfitTTM")), d["revenue"])
    d["gross_margin"]    = safe_div(safe_float(inc.get("grossProfit")), d["revenue"])
    d["operating_margin"]= safe_float(overview.get("OperatingMarginTTM"))
    d["net_margin"]      = safe_float(overview.get("ProfitMargin"))

    # ROIC approximation
    try:
        nopat = (d["net_income"] or 0) + (d["total_debt"] or 0) * 0.04
        invested = (d["total_equity"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0)
        d["roic"] = safe_div(nopat, invested)
    except Exception:
        d["roic"] = None

    # ── 1Y price history ──
    ts = daily.get("Time Series (Daily)", {})
    if ts:
        dates  = sorted(ts.keys())[-252:]
        prices = [safe_float(ts[d2]["4. close"]) for d2 in dates]
        d["price_history"] = pd.Series(prices, index=pd.to_datetime(dates), name="Price ($)")
    else:
        d["price_history"] = None

    return d


# ── Valuation computations ────────────────────────────────────────────────────

def compute_valuations(d, r, g, lifo_reserve=0.0):
    v = {}
    rg = r - g

    # ── Market ──
    v["market_cap"] = d["market_cap"] or \
                      ((d["price"] or 0) * (d["shares"] or 0) if d["price"] and d["shares"] else None)
    v["ev"]         = d["ev_dollars"] or \
                      ((v["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0))

    # ════════════════════════════════════════════════════════════
    # ELEMENT 1 — Asset Value (Reproduction Cost)
    # Most reliable: tangible current assets adjusted to today's cost
    # ════════════════════════════════════════════════════════════
    book_total = (d["book_value"] or 0) * (d["shares"] or 0) \
                 if d["book_value"] and d["shares"] else (d["total_equity"] or 0)
    # Add LIFO reserve (inventory adjustment to current cost)
    v["asset_value_total"]    = book_total + lifo_reserve
    v["asset_value_per_share"]= safe_div(v["asset_value_total"], d["shares"])

    # ════════════════════════════════════════════════════════════
    # ELEMENT 2 — Earnings Power Value (EPV)
    # Intermediate: value as earnings machine, zero growth assumed
    # EPV = Adjusted Earnings / R
    # Adjusted earnings: net income, backed out by maintenance capex proxy
    # ════════════════════════════════════════════════════════════
    net_inc = d["net_income"] or ((d["eps"] or 0) * (d["shares"] or 1))
    # Maintenance capex approximation: use reported capex (conservative)
    # Distributable cash flow = net income (AV does not provide D&A separately)
    adj_earnings       = net_inc
    v["adj_earnings"]  = adj_earnings
    v["epv_total"]     = safe_div(adj_earnings, r)
    v["epv_per_share"] = safe_div(v["epv_total"], d["shares"])

    # Franchise Value = EPV − Asset Value (confirms moat exists)
    v["franchise_value"]     = (v["epv_total"] or 0) - v["asset_value_total"] \
                               if v["epv_total"] else None
    v["franchise_per_share"] = safe_div(v["franchise_value"], d["shares"])
    v["has_franchise"]       = (v["franchise_value"] or 0) > 0

    # ════════════════════════════════════════════════════════════
    # ELEMENT 3 — Value of Growth
    # Lowest reliability: only valid when franchise confirmed (EPV > Asset Value)
    # PV = Capital × (ROC − G) / (R − G)
    # ════════════════════════════════════════════════════════════
    roc = d["roic"] or d["roe"] or 0
    v["roc_used"] = roc

    # Growth only creates value when ROC > R
    v["growth_adds_value"] = roc > r if roc else None

    # Growth Factor F = (ROC − G) / (R − G)
    v["growth_factor_F"] = safe_div(roc - g, rg) if rg > 0 and roc else None

    # Growth Multiplier M
    if r > 0 and roc and roc > 0:
        gr = g / r
        v["growth_mult_M"] = safe_div(1 - gr, 1 - gr * (r / roc))
    else:
        v["growth_mult_M"] = None

    # Full PV with growth = EPV × M  (only meaningful if franchise exists AND ROC > R)
    if v["epv_total"] and v["growth_mult_M"] and v["has_franchise"] and v["growth_adds_value"]:
        v["pv_with_growth"]          = v["epv_total"] * v["growth_mult_M"]
        v["pv_with_growth_per_share"]= safe_div(v["pv_with_growth"], d["shares"])
    else:
        v["pv_with_growth"]          = None
        v["pv_with_growth_per_share"]= None

    # ════════════════════════════════════════════════════════════
    # COMBINED INTRINSIC VALUE
    # Logic from the book:
    #   - No franchise (EPV ≤ Assets): use Asset Value
    #   - Franchise but ROC ≤ R: use EPV (growth neutral/destructive)
    #   - Franchise and ROC > R: use PV with growth
    # ════════════════════════════════════════════════════════════
    if not v["epv_total"]:
        v["intrinsic_value"]          = v["asset_value_total"]
        v["intrinsic_value_per_share"]= v["asset_value_per_share"]
        v["iv_basis"]                 = "Asset Value (no earnings data)"
    elif not v["has_franchise"]:
        v["intrinsic_value"]          = v["asset_value_total"]
        v["intrinsic_value_per_share"]= v["asset_value_per_share"]
        v["iv_basis"]                 = "Asset Value (EPV ≤ Assets — no moat)"
    elif not v["growth_adds_value"]:
        v["intrinsic_value"]          = v["epv_total"]
        v["intrinsic_value_per_share"]= v["epv_per_share"]
        v["iv_basis"]                 = "EPV (franchise exists but ROC ≤ R — growth neutral)"
    else:
        v["intrinsic_value"]          = v["pv_with_growth"]
        v["intrinsic_value_per_share"]= v["pv_with_growth_per_share"]
        v["iv_basis"]                 = "PV with Growth (franchise + ROC > R confirmed)"

    # ════════════════════════════════════════════════════════════
    # MARGIN OF SAFETY — against combined intrinsic value
    # ════════════════════════════════════════════════════════════
    iv = v["intrinsic_value"]
    iv_ps = v["intrinsic_value_per_share"]
    v["mos_per_share"] = (iv_ps - (d["price"] or 0)) if iv_ps and d["price"] else None
    v["mos_total"]     = (iv - (v["market_cap"] or 0)) if iv else None
    v["mos_pct"]       = safe_div(v["mos_total"], iv)

    # ── DDM ──
    if rg > 0 and d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        v["ddm_per_share"] = d["dividend_ttm"] / rg
        v["ddm_total"]     = v["ddm_per_share"] * (d["shares"] or 0)
    else:
        v["ddm_per_share"] = None
        v["ddm_total"]     = None

    # ── Cap Rate (Gabelli) ──
    op_cf = (d["ebitda"] or 0) - (d["capex"] or 0)
    v["cap_rate"]             = safe_div(op_cf, v["ev"]) if v["ev"] else None
    v["operating_cf_for_cap"] = op_cf

    # ── PEG ──
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    v["peg"] = safe_div(d["pe_ratio"], g_pct) if d["pe_ratio"] and g_pct else None

    # ── Sonkin adjusted P/E ──
    net_cash    = (d["cash"] or 0) - (d["total_debt"] or 0)
    int_on_cash = net_cash * 0.04 if net_cash > 0 else 0
    op_mktcap   = (v["market_cap"] or 0) - net_cash
    op_earn     = (d["net_income"] or 0) - int_on_cash
    v["sonkin_pe"] = safe_div(op_mktcap, op_earn) if op_earn and op_mktcap > 0 else None
    v["net_cash"]  = net_cash

    # ── 1Y total return ──
    v["total_return"] = None
    if d["price_history"] is not None and len(d["price_history"]) > 1:
        p0 = d["price_history"].iloc[0]
        p1 = d["price_history"].iloc[-1]
        v["total_return"] = safe_div(p1 - p0 + (d["dividend_ttm"] or 0), p0)

    return v


# ── Signal helpers ────────────────────────────────────────────────────────────

def mos_signal(mos_pct):
    if mos_pct is None: return signal_html("N/A", "signal-na")
    if mos_pct >= 0.33: return signal_html(f"▲ BUY  ({fmt_pct(mos_pct)} MoS)", "signal-buy")
    if mos_pct >= 0.10: return signal_html(f"◆ HOLD ({fmt_pct(mos_pct)} MoS)", "signal-hold")
    return signal_html(f"▼ EXPENSIVE ({fmt_pct(mos_pct)} MoS)", "signal-sell")

def cap_signal(cap_rate):
    if cap_rate is None: return signal_html("N/A", "signal-na")
    if cap_rate >= 0.08: return signal_html(f"▲ ATTRACTIVE ({fmt_pct(cap_rate)})", "signal-buy")
    if cap_rate >= 0.05: return signal_html(f"◆ FAIR ({fmt_pct(cap_rate)})", "signal-hold")
    return signal_html(f"▼ EXPENSIVE ({fmt_pct(cap_rate)})", "signal-sell")

def peg_signal(peg):
    if peg is None: return signal_html("N/A", "signal-na")
    if peg < 1.0:   return signal_html(f"▲ UNDERVALUED ({fmt_x(peg)})", "signal-buy")
    if peg < 3.0:   return signal_html(f"◆ FAIR ({fmt_x(peg)})", "signal-hold")
    return signal_html(f"▼ OVERVALUED ({fmt_x(peg)})", "signal-sell")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Value Investing Toolkit")
    st.caption("Hooke · Graham · Greenwald methodology")
    st.divider()

    api_key = st.text_input(
        "Alpha Vantage API Key",
        type="password",
        help="Free key at alphavantage.co/support/#api-key"
    )
    if not api_key:
        st.info("Get a free API key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("#### Assumptions")
    r = st.slider("Cost of Capital (R)", 0.04, 0.20, 0.09, 0.005, format="%.3f")
    g = st.slider("Perpetual Growth Rate (G)", 0.00, 0.08, 0.03, 0.005, format="%.3f")
    lifo_reserve = st.number_input(
        "LIFO Reserve ($M)", min_value=0.0, value=0.0, step=10.0, format="%.1f"
    ) * 1e6

    st.divider()
    st.caption("Free tier: 25 requests/day · 5/min")
    run = st.button("🔍 Analyse", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 Value Investing Toolkit")

if g >= r:
    st.error("⚠️ Growth rate G must be less than Cost of Capital R.")
    st.stop()

if not api_key:
    st.warning("Enter your Alpha Vantage API key in the sidebar to begin.")
    st.stop()

if not run and "last_ticker" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click **Analyse**.")
    st.stop()

ticker = ticker_input if run else st.session_state.get("last_ticker", ticker_input)
if run:
    st.session_state["last_ticker"] = ticker_input
    st.cache_data.clear()

# ── Fetch ──
with st.spinner(f"Fetching data for **{ticker}** from Alpha Vantage (pacing requests, ~8 sec)…"):
    try:
        overview, income, balance, cashflow, quote, daily = fetch_all(ticker, api_key)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Network error: {e}")
        st.stop()

if not overview or not overview.get("Symbol"):
    st.error(f"No data returned for **{ticker}**. Check the ticker and your API key.")
    st.stop()

d = extract_financials(overview, income, balance, cashflow, quote, daily)
v = compute_valuations(d, r, g, lifo_reserve)

# ── Company header ────────────────────────────────────────────────────────────
col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
with col_a:
    st.subheader(f"{d['name']}  [{ticker}]")
    st.caption(f"{d['sector']}  ·  {d['industry']}")
with col_b:
    st.metric("Price", f"${d['price']:,.2f}" if d["price"] else "N/A")
with col_c:
    st.metric("Market Cap", fmt_currency(v["market_cap"]))
with col_d:
    st.metric("1Y Total Return", fmt_pct(v["total_return"]) if v["total_return"] else "N/A")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Overview", "💰 Valuation", "🚀 Growth", "🏭 Asset-Based", "🎯 Summary"
])

# ══ TAB 1 ═════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enterprise Value",  fmt_currency(v["ev"]))
    c2.metric("P/E Ratio",         fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A")
    c3.metric("P/B Ratio",         fmt_x(d["pb_ratio"]) if d["pb_ratio"] else "N/A")
    c4.metric("P/S Ratio",         fmt_x(d["ps_ratio"]) if d["ps_ratio"] else "N/A")

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Beta",             f"{d['beta']:.2f}" if d["beta"] else "N/A")
    c2b.metric("Dividend Yield",   fmt_pct(d["dividend_yield"]) if d["dividend_yield"] else "0.0%")
    c3b.metric("1Y Total Return",  fmt_pct(v["total_return"]) if v["total_return"] else "N/A")
    c4b.metric("EPS (TTM)",        f"${d['eps']:,.2f}" if d["eps"] else "N/A")

    st.markdown('<div class="section-header">Price Chart (1 Year)</div>', unsafe_allow_html=True)
    if d["price_history"] is not None:
        st.line_chart(d["price_history"], use_container_width=True)
    else:
        st.info("Price history unavailable.")

    st.markdown('<div class="section-header">Profitability</div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Gross Margin",      fmt_pct(d["gross_margin"]))
    pc2.metric("Operating Margin",  fmt_pct(d["operating_margin"]))
    pc3.metric("Net Margin",        fmt_pct(d["net_margin"]))
    pc4.metric("ROE",               fmt_pct(d["roe"]))

# ══ TAB 2 ═════════════════════════════════════════════════════════════════════
with tab2:

    # ── Combined Intrinsic Value banner ──
    st.markdown('<div class="section-header">Combined Intrinsic Value — Greenwald Three-Element Framework</div>', unsafe_allow_html=True)

    iv_ps = v["intrinsic_value_per_share"]
    price  = d["price"]

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Intrinsic Value / Share", f"${iv_ps:,.2f}" if iv_ps else "N/A")
    b2.metric("Current Price",           f"${price:,.2f}" if price else "N/A")
    b3.metric("Margin of Safety",        fmt_pct(v["mos_pct"]) if v["mos_pct"] else "N/A")
    b4.metric("IV Basis",                v.get("iv_basis", "N/A")[:30] + "…"
                                         if len(v.get("iv_basis","")) > 30 else v.get("iv_basis","N/A"))

    st.markdown("**Signal:** " + mos_signal(v["mos_pct"]), unsafe_allow_html=True)
    st.caption(f"IV basis: {v.get('iv_basis','N/A')}")
    st.info("Graham: ≥33% margin of safety for adequate protection; ≥50% for strong buy.")

    st.divider()

    # ── Element 1: Asset Value ──
    st.markdown('<div class="section-header">Element 1 — Asset Value (Reproduction Cost) · Highest Reliability</div>', unsafe_allow_html=True)
    st.caption("What would it cost a competitor to build this business from scratch? Based on book value + adjustments.")

    a1, a2, a3 = st.columns(3)
    a1.metric("Book Equity (Total)",     fmt_currency((d["book_value"] or 0) * (d["shares"] or 0)
                                         if d["book_value"] and d["shares"] else d["total_equity"]))
    a2.metric("LIFO Reserve Added",      fmt_currency(lifo_reserve))
    a3.metric("Asset Value (Total)",     fmt_currency(v["asset_value_total"]))

    a4, a5 = st.columns(2)
    a4.metric("Asset Value / Share",     f"${v['asset_value_per_share']:,.2f}"
                                         if v["asset_value_per_share"] else "N/A")
    a5.metric("Price vs Asset Value",
              f"{safe_div(price, v['asset_value_per_share']):.1f}×"
              if price and v["asset_value_per_share"] else "N/A",
              help="Price / Asset Value per share. <1.0 = trading below reproduction cost.")

    st.divider()

    # ── Element 2: EPV ──
    st.markdown('<div class="section-header">Element 2 — Earnings Power Value (EPV) · Intermediate Reliability</div>', unsafe_allow_html=True)
    st.caption("Value as an earnings machine assuming zero future growth. EPV = Adjusted Earnings ÷ R")

    e1, e2, e3 = st.columns(3)
    e1.metric("Adjusted Earnings (TTM)", fmt_currency(v["adj_earnings"]))
    e2.metric("EPV (Total)",             fmt_currency(v["epv_total"]))
    e3.metric("EPV / Share",             f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")

    # Franchise value
    fv = v["franchise_value"]
    fv_ps = v["franchise_per_share"]
    fv_pct = safe_div(fv, v["epv_total"]) if v["epv_total"] else None

    e4, e5, e6 = st.columns(3)
    e4.metric("Asset Value (Total)",     fmt_currency(v["asset_value_total"]))
    e5.metric("Franchise Value",         fmt_currency(fv),
              help="EPV − Asset Value. Positive = moat exists.")
    e6.metric("Franchise / Share",       f"${fv_ps:,.2f}" if fv_ps else "N/A")

    if fv is not None:
        if fv > 0:
            st.success(f"✅ EPV (${v['epv_per_share']:,.0f}) > Asset Value (${v['asset_value_per_share']:,.0f}) — franchise confirmed. Moat = {fmt_pct(fv_pct)} of EPV.")
        else:
            st.warning("⚠️ EPV ≤ Asset Value — no franchise value detected. Use Asset Value as intrinsic value.")

    st.divider()

    # ── Element 3: Growth Value ──
    st.markdown('<div class="section-header">Element 3 — Value of Growth · Lowest Reliability</div>', unsafe_allow_html=True)
    st.caption("Only added when franchise is confirmed AND ROC > R. PV = Capital × (ROC − G) ÷ (R − G)")

    roc = v["roc_used"]
    g3a, g3b, g3c, g3d = st.columns(4)
    g3a.metric("ROC (ROIC/ROE)",   fmt_pct(roc) if roc else "N/A")
    g3b.metric("Cost of Capital R", fmt_pct(r))
    g3c.metric("Growth Factor F",  fmt_x(v["growth_factor_F"]))
    g3d.metric("PV with Growth / Share",
               f"${v['pv_with_growth_per_share']:,.2f}" if v["pv_with_growth_per_share"] else "N/A")

    if not v["has_franchise"]:
        st.error("🔴 No franchise confirmed — growth value not added to intrinsic value.")
    elif not v["growth_adds_value"]:
        st.warning(f"⚠️ ROC ({fmt_pct(roc)}) ≤ R ({fmt_pct(r)}) — growth is value-neutral or destructive. EPV used as intrinsic value.")
    else:
        st.success(f"✅ Franchise confirmed + ROC ({fmt_pct(roc)}) > R ({fmt_pct(r)}) — growth adds value. Full PV = ${v['pv_with_growth_per_share']:,.2f}/share.")

    st.divider()

    # ── DDM ──
    st.markdown('<div class="section-header">Dividend Discount Model (DDM) — Ch.6</div>', unsafe_allow_html=True)
    st.caption("V = Dividend ÷ (R − G) — applicable only to dividend-paying stocks.")
    if d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        d1, d2, d3 = st.columns(3)
        d1.metric("Dividend / Share (TTM)", f"${d['dividend_ttm']:,.4f}")
        d2.metric("DDM Value / Share",      f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A")
        d3.metric("DDM vs Price",
                  f"${(v['ddm_per_share'] or 0) - (d['price'] or 0):+,.2f}"
                  if v["ddm_per_share"] and d["price"] else "N/A")
    else:
        st.warning("No dividends paid — DDM not applicable.")

# ══ TAB 3 ═════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Growth Factor F — Ch.7</div>', unsafe_allow_html=True)
    st.caption("F = (ROC − G) ÷ (R − G).  If ROC = R, growth adds zero value.")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROIC (approx.)",  fmt_pct(d["roic"]))
    g2.metric("ROE",             fmt_pct(d["roe"]))
    g3.metric("Growth Factor F", fmt_x(v["growth_factor_F"]))
    g4.metric("Growth Mult. M",  fmt_x(v["growth_mult_M"]))

    roc_used = d["roic"] or d["roe"]
    if roc_used and v["growth_factor_F"]:
        if roc_used > r:
            st.success(f"✅ ROC ({fmt_pct(roc_used)}) > R ({fmt_pct(r)}) — growth creates value (F = {fmt_x(v['growth_factor_F'])}).")
        elif abs(roc_used - r) < 0.01:
            st.warning("⚠️ ROC ≈ R — growth is value-neutral.")
        else:
            st.error(f"🔴 ROC ({fmt_pct(roc_used)}) < R ({fmt_pct(r)}) — growth destroys value.")
    else:
        st.info("ROIC/ROE not available.")

    st.markdown('<div class="section-header">PEG Ratio — Ch.11 (Glenn Greenberg)</div>', unsafe_allow_html=True)
    st.caption("PEG = P/E ÷ Growth Rate (%).  < 1.0 = undervalued relative to growth.")

    pg1, pg2, pg3 = st.columns(3)
    pg1.metric("P/E Ratio",       fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A")
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    pg2.metric("Growth Rate Used", f"{g_pct:.1f}%")
    pg3.metric("PEG Ratio",        fmt_x(v["peg"]) if v["peg"] else "N/A")
    st.markdown("**Signal:** " + peg_signal(v["peg"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Revenue & Earnings Trends</div>', unsafe_allow_html=True)
    tr1, tr2, tr3 = st.columns(3)
    tr1.metric("Revenue Growth (YOY)",  fmt_pct(d["revenue_growth"]))
    tr2.metric("Earnings Growth (YOY)", fmt_pct(d["earnings_growth"]))
    tr3.metric("Revenue (TTM)",         fmt_currency(d["revenue"]))

# ══ TAB 4 ═════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Franchise Value — Ch.5 (Greenwald)</div>', unsafe_allow_html=True)
    st.caption("Franchise Value = EPV − Reproduction Cost of Assets")

    fv1, fv2, fv3 = st.columns(3)
    fv1.metric("EPV (Total)",        fmt_currency(v["epv_total"]))
    fv2.metric("Asset Value (Repro. Cost)", fmt_currency(v["asset_value_total"]),
               help="Total equity book value + LIFO reserve")
    fv3.metric("Franchise Value",    fmt_currency(v["franchise_value"]))

    if v["franchise_value"] and v["epv_total"]:
        fv_pct = safe_div(v["franchise_value"], v["epv_total"])
        if v["franchise_value"] > 0:
            st.success(f"✅ Positive franchise value ({fmt_pct(fv_pct)} of EPV) — competitive moat likely.")
        else:
            st.warning("⚠️ EPV ≤ Asset Value — limited evidence of a durable moat.")

    st.markdown('<div class="section-header">Cap Rate — Ch.10 (Mario Gabelli)</div>', unsafe_allow_html=True)
    st.caption("Cap Rate = (EBITDA − Capex) ÷ Enterprise Value")

    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("EBITDA − Capex",   fmt_currency(v["operating_cf_for_cap"]))
    ca2.metric("Enterprise Value", fmt_currency(v["ev"]))
    ca3.metric("Cap Rate",         fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A")
    st.markdown("**Signal:** " + cap_signal(v["cap_rate"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Adjusted P/E — Ch.16 (Paul Sonkin)</div>', unsafe_allow_html=True)
    st.caption("Strips net cash to reveal the true operating earnings multiple.")

    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Net Cash",         fmt_currency(v["net_cash"]), help="Cash − Total Debt")
    sp2.metric("Op. Market Cap",   fmt_currency((v["market_cap"] or 0) - v["net_cash"]))
    sp3.metric("Sonkin Adj. P/E",  fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A")

# ══ TAB 5 ═════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 🎯 Summary: {d['name']} [{ticker}]")
    st.caption(f"R = {fmt_pct(r)}  |  G = {fmt_pct(g)}  |  {pd.Timestamp.now().strftime('%d %b %Y')}")

    summary_rows = [
        ("Market",     "Price",                  f"${d['price']:,.2f}" if d["price"] else "N/A",   "—"),
        ("Market",     "Market Cap",              fmt_currency(v["market_cap"]),                    "—"),
        ("Market",     "Enterprise Value",        fmt_currency(v["ev"]),                            "—"),
        ("Market",     "1Y Total Return",         fmt_pct(v["total_return"]) if v["total_return"] else "N/A", "—"),
        ("Intrinsic",  "Asset Value / Share",     f"${v['asset_value_per_share']:,.2f}" if v["asset_value_per_share"] else "N/A", "—"),
        ("Intrinsic",  "EPV / Share",             f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A", "—"),
        ("Intrinsic",  "Franchise Value",         fmt_currency(v["franchise_value"]),
                                                  "✅ Moat" if v["has_franchise"] else "⚠️ No moat"),
        ("Intrinsic",  "PV with Growth / Share",  f"${v['pv_with_growth_per_share']:,.2f}" if v["pv_with_growth_per_share"] else "N/A",
                                                  "✅ Valid" if v["pv_with_growth_per_share"] else "⚠️ Not used"),
        ("Intrinsic",  "IV Basis",                v.get("iv_basis","N/A")[:40], "—"),
        ("Valuation",  "Margin of Safety",        fmt_pct(v["mos_pct"]) if v["mos_pct"] else "N/A",
                                                  "✅ BUY" if (v["mos_pct"] or 0) >= 0.33
                                                  else ("⚠️ HOLD" if (v["mos_pct"] or 0) >= 0.10 else "🔴 SELL")),
        ("Valuation",  "DDM / Share",             f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A", "—"),
        ("Valuation",  "P/E Ratio",               fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A", "—"),
        ("Growth",     "ROIC",                    fmt_pct(d["roic"]) if d["roic"] else "N/A",      "—"),
        ("Growth",     "Growth adds value?",      "Yes" if v["growth_adds_value"] else "No",
                                                  "✅ Yes" if v["growth_adds_value"] else "🔴 No"),
        ("Growth",     "Growth Factor F",         fmt_x(v["growth_factor_F"]) if v["growth_factor_F"] else "N/A",
                                                  "✅ F>1" if (v["growth_factor_F"] or 0) > 1 else "⚠️ F≤1"),
        ("Growth",     "PEG Ratio",               fmt_x(v["peg"]) if v["peg"] else "N/A",
                                                  "✅ <1" if v["peg"] and v["peg"] < 1
                                                  else ("⚠️ 1–3" if v["peg"] and v["peg"] < 3 else "🔴 >3")),
        ("Asset",      "Cap Rate",                fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A",
                                                  "✅ >8%" if (v["cap_rate"] or 0) >= 0.08
                                                  else ("⚠️ 5–8%" if (v["cap_rate"] or 0) >= 0.05 else "🔴 <5%")),
        ("Asset",      "Sonkin Adj. P/E",         fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A", "—"),
    ]

    df_summary = pd.DataFrame(summary_rows, columns=["Category", "Metric", "Value", "Signal"])

    def style_signal(val):
        if "✅" in str(val): return "color: #00e676; font-weight: 600"
        if "⚠️" in str(val): return "color: #ffd740; font-weight: 600"
        if "🔴" in str(val): return "color: #ff5252; font-weight: 600"
        return "color: #888"

    styled = df_summary.style.map(style_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=530)

    st.divider()
    buy_s  = sum(1 for *_, sig in summary_rows if "✅" in sig)
    warn_s = sum(1 for *_, sig in summary_rows if "⚠️" in sig)
    sell_s = sum(1 for *_, sig in summary_rows if "🔴" in sig)

    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("✅ Buy Signals",  buy_s)
    vc2.metric("⚠️ Hold Signals", warn_s)
    vc3.metric("🔴 Sell Signals", sell_s)

    if buy_s > sell_s and buy_s >= 3:
        vc4.success("🟢 OVERALL: ATTRACTIVE")
    elif sell_s >= 3:
        vc4.error("🔴 OVERALL: EXPENSIVE")
    else:
        vc4.warning("🟡 OVERALL: MIXED / REVIEW")

    st.caption("⚠️ Research only — not financial advice. Verify data independently.")
