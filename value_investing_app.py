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
    overview   = av_get("OVERVIEW",             ticker, api_key)
    income     = av_get("INCOME_STATEMENT",     ticker, api_key)
    balance    = av_get("BALANCE_SHEET",        ticker, api_key)
    cashflow   = av_get("CASH_FLOW",            ticker, api_key)
    quote      = av_get("GLOBAL_QUOTE",         ticker, api_key)
    daily      = av_get("TIME_SERIES_DAILY",    ticker, api_key, outputsize="compact")
    return overview, income, balance, cashflow, quote, daily


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

    v["market_cap"]  = d["market_cap"] or \
                       ((d["price"] or 0) * (d["shares"] or 0) if d["price"] and d["shares"] else None)
    v["ev"]          = d["ev_dollars"] or \
                       ((v["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0))

    # EPV
    adj_earn         = d["net_income"] or ((d["eps"] or 0) * (d["shares"] or 1))
    v["epv_total"]   = safe_div(adj_earn, r)
    v["epv_per_share"]= safe_div(v["epv_total"], d["shares"])

    # Reproduction cost
    book_total       = (d["book_value"] or 0) * (d["shares"] or 0) \
                       if d["book_value"] and d["shares"] else (d["total_equity"] or 0)
    v["reproduction_cost"] = book_total + lifo_reserve

    # Franchise value
    v["franchise_value"]    = (v["epv_total"] or 0) - v["reproduction_cost"] if v["epv_total"] else None
    v["franchise_per_share"]= safe_div(v["franchise_value"], d["shares"])

    # Margin of Safety
    v["mos_per_share"] = ((v["epv_per_share"] or 0) - (d["price"] or 0)) \
                         if v["epv_per_share"] and d["price"] else None
    v["mos_total"]     = (v["epv_total"] or 0) - (v["market_cap"] or 0) if v["epv_total"] else None
    v["mos_pct"]       = safe_div(v["mos_total"], v["epv_total"])

    # DDM
    if rg > 0 and d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        v["ddm_per_share"] = d["dividend_ttm"] / rg
        v["ddm_total"]     = v["ddm_per_share"] * (d["shares"] or 0)
    else:
        v["ddm_per_share"] = None
        v["ddm_total"]     = None

    # Growth Factor F
    roc = d["roic"] or d["roe"] or 0
    v["growth_factor_F"] = safe_div(roc - g, rg) if rg > 0 and roc else None

    # Growth Multiplier M
    if r > 0 and roc and roc > 0:
        gr = g / r
        v["growth_mult_M"] = safe_div(1 - gr, 1 - gr * (r / roc))
    else:
        v["growth_mult_M"] = None

    # Cap Rate
    op_cf = (d["ebitda"] or 0) - (d["capex"] or 0)
    v["cap_rate"]              = safe_div(op_cf, v["ev"]) if v["ev"] else None
    v["operating_cf_for_cap"]  = op_cf

    # PEG
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    v["peg"]  = safe_div(d["pe_ratio"], g_pct) if d["pe_ratio"] and g_pct else None

    # Sonkin adjusted P/E
    net_cash     = (d["cash"] or 0) - (d["total_debt"] or 0)
    int_on_cash  = net_cash * 0.04 if net_cash > 0 else 0
    op_mktcap    = (v["market_cap"] or 0) - net_cash
    op_earn      = (d["net_income"] or 0) - int_on_cash
    v["sonkin_pe"] = safe_div(op_mktcap, op_earn) if op_earn and op_mktcap > 0 else None
    v["net_cash"]  = net_cash

    # 1Y total return
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
with st.spinner(f"Fetching data for **{ticker}** from Alpha Vantage…"):
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
    st.markdown('<div class="section-header">Earnings Power Value (EPV) — Greenwald</div>', unsafe_allow_html=True)
    st.caption("EPV = Adjusted Earnings ÷ R  (zero-growth, most conservative)")

    e1, e2, e3 = st.columns(3)
    e1.metric("EPV (Total)",     fmt_currency(v["epv_total"]))
    e2.metric("EPV per Share",   f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    e3.metric("Current Price",   f"${d['price']:,.2f}" if d["price"] else "N/A")

    st.markdown('<div class="section-header">Margin of Safety</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("MoS per Share",  f"${v['mos_per_share']:,.2f}" if v["mos_per_share"] else "N/A",
              delta=fmt_pct(v["mos_pct"]) if v["mos_pct"] else None)
    m2.metric("MoS Total",      fmt_currency(v["mos_total"]))
    m3.metric("MoS %",          fmt_pct(v["mos_pct"]))
    st.markdown("**Signal:** " + mos_signal(v["mos_pct"]), unsafe_allow_html=True)
    st.info("Graham: ≥33% margin of safety for adequate protection; ≥50% for strong buy.")

    st.markdown('<div class="section-header">Dividend Discount Model (DDM) — Ch.6</div>', unsafe_allow_html=True)
    st.caption("V = Dividend ÷ (R − G)")
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
    fv2.metric("Reproduction Cost",  fmt_currency(v["reproduction_cost"]),
               help="Total equity book value + LIFO reserve")
    fv3.metric("Franchise Value",    fmt_currency(v["franchise_value"]))

    if v["franchise_value"] and v["epv_total"]:
        fv_pct = safe_div(v["franchise_value"], v["epv_total"])
        if v["franchise_value"] > 0:
            st.success(f"✅ Positive franchise value ({fmt_pct(fv_pct)} of EPV) — competitive moat likely.")
        else:
            st.warning("⚠️ EPV ≤ Reproduction Cost — limited evidence of a durable moat.")

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
        ("Market",    "Price",               f"${d['price']:,.2f}" if d["price"] else "N/A",   "—"),
        ("Market",    "Market Cap",           fmt_currency(v["market_cap"]),                    "—"),
        ("Market",    "Enterprise Value",     fmt_currency(v["ev"]),                            "—"),
        ("Market",    "1Y Total Return",      fmt_pct(v["total_return"]) if v["total_return"] else "N/A", "—"),
        ("Valuation", "EPV / Share",          f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A", "—"),
        ("Valuation", "Margin of Safety",     fmt_pct(v["mos_pct"]) if v["mos_pct"] else "N/A",
                                              "✅ BUY" if (v["mos_pct"] or 0) >= 0.33
                                              else ("⚠️ HOLD" if (v["mos_pct"] or 0) >= 0.10 else "🔴 SELL")),
        ("Valuation", "DDM / Share",          f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A", "—"),
        ("Valuation", "P/E Ratio",            fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A", "—"),
        ("Growth",    "ROIC",                 fmt_pct(d["roic"]) if d["roic"] else "N/A",      "—"),
        ("Growth",    "Growth Factor F",      fmt_x(v["growth_factor_F"]) if v["growth_factor_F"] else "N/A",
                                              "✅ Value" if (v["growth_factor_F"] or 0) > 1 else "⚠️ Neutral"),
        ("Growth",    "PEG Ratio",            fmt_x(v["peg"]) if v["peg"] else "N/A",
                                              "✅ <1" if v["peg"] and v["peg"] < 1
                                              else ("⚠️ 1–3" if v["peg"] and v["peg"] < 3 else "🔴 >3")),
        ("Asset",     "Franchise Value",      fmt_currency(v["franchise_value"]),
                                              "✅ Moat" if (v["franchise_value"] or 0) > 0 else "⚠️ None"),
        ("Asset",     "Cap Rate",             fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A",
                                              "✅ >8%" if (v["cap_rate"] or 0) >= 0.08
                                              else ("⚠️ 5–8%" if (v["cap_rate"] or 0) >= 0.05 else "🔴 <5%")),
        ("Asset",     "Sonkin Adj. P/E",      fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A", "—"),
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
  .metric-sub   { font-size: 11px; color: #666; margin-top: 2px; }

  .signal-buy    { color: #00e676; font-weight: 600; }
  .signal-sell   { color: #ff5252; font-weight: 600; }
  .signal-hold   { color: #ffd740; font-weight: 600; }
  .signal-na     { color: #888; }

  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #2d2d2d;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
  }

  .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
  .stAlert { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_currency(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
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

def signal_html(label, colour_class):
    return f'<span class="{colour_class}">{label}</span>'

def safe_div(a, b):
    try:
        if b == 0 or b is None or np.isnan(b):
            return None
        return a / b
    except Exception:
        return None


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info

    # Income statement
    try:
        inc = t.financials  # columns = fiscal year ends
    except Exception:
        inc = pd.DataFrame()

    # Balance sheet
    try:
        bal = t.balance_sheet
    except Exception:
        bal = pd.DataFrame()

    # Cash flow
    try:
        cf = t.cashflow
    except Exception:
        cf = pd.DataFrame()

    # Historical prices — 1 year
    try:
        hist = t.history(period="1y")
    except Exception:
        hist = pd.DataFrame()

    # Dividends trailing 12m
    try:
        divs = t.dividends
        ttm_div = float(divs.last("365D").sum()) if len(divs) else 0.0
    except Exception:
        ttm_div = 0.0

    return info, inc, bal, cf, hist, ttm_div


def extract_financials(info, inc, bal, cf, ttm_div):
    """Pull the key numbers we need, with graceful fallbacks."""
    d = {}

    # ── Market data ──
    d["price"]       = info.get("currentPrice") or info.get("regularMarketPrice")
    d["shares"]      = info.get("sharesOutstanding")
    d["market_cap"]  = info.get("marketCap")
    d["beta"]        = info.get("beta")
    d["sector"]      = info.get("sector", "N/A")
    d["industry"]    = info.get("industryDisp") or info.get("industry", "N/A")
    d["name"]        = info.get("longName", info.get("shortName", ""))

    # ── P&L ──
    d["revenue"]     = info.get("totalRevenue")
    d["net_income"]  = info.get("netIncomeToCommon")
    d["ebitda"]      = info.get("ebitda")
    d["eps"]         = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")

    # ── Balance sheet ──
    d["total_debt"]       = info.get("totalDebt", 0) or 0
    d["cash"]             = info.get("totalCash", 0) or 0
    d["book_value"]       = info.get("bookValue")         # per share
    d["total_assets"]     = info.get("totalAssets")
    d["total_equity"]     = info.get("totalStockholderEquity") or info.get("bookValue")

    # Try balance sheet rows directly
    for row_name in ["Total Stockholder Equity", "Stockholders Equity", "Common Stock Equity"]:
        if not d["total_equity"] and bal is not None and row_name in bal.index:
            d["total_equity"] = float(bal.loc[row_name].iloc[0])
            break

    # ── Cash flow ──
    d["capex"] = info.get("capitalExpenditures", 0) or 0
    if d["capex"] > 0:
        d["capex"] = -d["capex"]   # yfinance returns negative; normalise to positive

    d["operating_cf"] = info.get("operatingCashflow", 0) or 0
    d["free_cf"]      = info.get("freeCashflow", 0) or 0

    # ── Dividends ──
    d["dividend_ttm"] = ttm_div
    d["dividend_yield"] = info.get("dividendYield", 0) or 0

    # ── Multiples ──
    d["pe_ratio"]    = info.get("trailingPE") or info.get("forwardPE")
    d["pb_ratio"]    = info.get("priceToBook")
    d["ps_ratio"]    = info.get("priceToSalesTrailingTwelveMonths")
    d["ev"]          = info.get("enterpriseValue")

    # Recalc EV if missing
    if not d["ev"] and d["market_cap"]:
        d["ev"] = (d["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0)

    # ── Growth ──
    d["revenue_growth"]  = info.get("revenueGrowth")
    d["earnings_growth"] = info.get("earningsGrowth")
    d["eps_growth_5y"]   = info.get("earningsQuarterlyGrowth")   # proxy

    # ── Profitability ──
    d["roe"] = info.get("returnOnEquity")
    d["roa"] = info.get("returnOnAssets")
    d["roic"] = None  # computed below
    d["gross_margin"]  = info.get("grossMargins")
    d["operating_margin"] = info.get("operatingMargins")
    d["net_margin"]    = info.get("profitMargins")

    # ROIC approximation: NOPAT / Invested Capital
    try:
        nopat = (d["net_income"] or 0) + (d["total_debt"] or 0) * 0.04   # rough tax-adjusted interest add-back
        invested_cap = (d["total_equity"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0)
        d["roic"] = safe_div(nopat, invested_cap)
    except Exception:
        pass

    return d


# ── Formula computations ──────────────────────────────────────────────────────

def compute_valuations(d, r, g, lifo_reserve=0.0):
    v = {}

    # 1. Market Cap (sanity check)
    v["market_cap"] = (d["shares"] or 0) * (d["price"] or 0) if d["shares"] and d["price"] else d["market_cap"]

    # 2. Enterprise Value
    v["ev"] = (v["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0)

    # 3. Total Return (1Y) — computed in chart section from hist
    v["total_return"] = None   # filled from price history

    # 4. Earnings Power Value  EPV = Adjusted Earnings / R
    adj_earnings = d["net_income"] or (d["eps"] or 0) * (d["shares"] or 1)
    v["epv_total"]  = safe_div(adj_earnings, r)
    v["epv_per_share"] = safe_div(v["epv_total"], d["shares"]) if d["shares"] else None

    # 5. Reproduction Cost of Assets (simplified: book value + LIFO reserve)
    book_total = (d["book_value"] or 0) * (d["shares"] or 0) if d["book_value"] and d["shares"] else (d["total_equity"] or 0)
    v["reproduction_cost"] = book_total + lifo_reserve

    # 6. Franchise Value = EPV - Reproduction Cost
    v["franchise_value"] = (v["epv_total"] or 0) - v["reproduction_cost"] if v["epv_total"] else None
    v["franchise_per_share"] = safe_div(v["franchise_value"], d["shares"]) if d["shares"] else None

    # 7. Margin of Safety  (using EPV as intrinsic value proxy)
    v["mos_total"]     = (v["epv_total"] or 0) - (v["market_cap"] or 0) if v["epv_total"] else None
    v["mos_per_share"] = (v["epv_per_share"] or 0) - (d["price"] or 0) if v["epv_per_share"] and d["price"] else None
    v["mos_pct"]       = safe_div(v["mos_total"], v["epv_total"]) if v["epv_total"] else None

    # 8. DDM  V = Dividend / (R - G)
    rg = r - g
    if rg > 0 and d["dividend_ttm"]:
        v["ddm_per_share"] = d["dividend_ttm"] / rg
        v["ddm_total"]     = v["ddm_per_share"] * (d["shares"] or 0)
    else:
        v["ddm_per_share"] = None
        v["ddm_total"]     = None

    # 9. Growth Factor F = (ROC - G) / (R - G)
    roc = d["roic"] or d["roe"] or 0
    if rg > 0 and roc:
        v["growth_factor_F"] = safe_div(roc - g, rg)
    else:
        v["growth_factor_F"] = None

    # 10. Growth Multiplier M  =  (1 - G/R) / (1 - (G/R)(R/ROC))
    if r > 0 and roc and roc > 0:
        gr_ratio = g / r
        v["growth_mult_M"] = safe_div(1 - gr_ratio, 1 - gr_ratio * (r / roc))
    else:
        v["growth_mult_M"] = None

    # 11. Cap Rate = Operating CF / EV
    op_cf = (d["ebitda"] or 0) - (d["capex"] or 0)
    v["cap_rate"] = safe_div(op_cf, v["ev"]) if v["ev"] else None
    v["operating_cf_for_cap"] = op_cf

    # 12. PEG Ratio
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    v["peg"] = safe_div(d["pe_ratio"], g_pct) if d["pe_ratio"] and g_pct else None

    # 13. Adjusted P/E (Sonkin) — strip out net cash
    net_cash = (d["cash"] or 0) - (d["total_debt"] or 0)
    interest_on_cash = net_cash * 0.04 if net_cash > 0 else 0
    op_market_cap = (v["market_cap"] or 0) - net_cash
    op_earnings   = (d["net_income"] or 0) - interest_on_cash
    v["sonkin_pe"] = safe_div(op_market_cap, op_earnings) if op_earnings and op_market_cap > 0 else None
    v["net_cash"]  = net_cash

    # 14. Relative Strength vs SPY (filled from chart section)
    v["relative_strength"] = None

    return v


def mos_signal(mos_pct):
    if mos_pct is None:
        return signal_html("N/A", "signal-na")
    if mos_pct >= 0.33:
        return signal_html(f"▲ BUY  ({fmt_pct(mos_pct)} MoS)", "signal-buy")
    if mos_pct >= 0.10:
        return signal_html(f"◆ HOLD ({fmt_pct(mos_pct)} MoS)", "signal-hold")
    return signal_html(f"▼ EXPENSIVE ({fmt_pct(mos_pct)} MoS)", "signal-sell")

def cap_signal(cap_rate):
    if cap_rate is None:
        return signal_html("N/A", "signal-na")
    if cap_rate >= 0.08:
        return signal_html(f"▲ ATTRACTIVE ({fmt_pct(cap_rate)})", "signal-buy")
    if cap_rate >= 0.05:
        return signal_html(f"◆ FAIR ({fmt_pct(cap_rate)})", "signal-hold")
    return signal_html(f"▼ EXPENSIVE ({fmt_pct(cap_rate)})", "signal-sell")

def peg_signal(peg):
    if peg is None:
        return signal_html("N/A", "signal-na")
    if peg < 1.0:
        return signal_html(f"▲ UNDERVALUED ({fmt_x(peg)})", "signal-buy")
    if peg < 3.0:
        return signal_html(f"◆ FAIR ({fmt_x(peg)})", "signal-hold")
    return signal_html(f"▼ OVERVALUED ({fmt_x(peg)})", "signal-sell")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Value Investing Toolkit")
    st.caption("Hooke • Graham • Greenwald methodology")
    st.divider()

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown('<div class="section-header">Assumptions</div>', unsafe_allow_html=True)
    r = st.slider("Cost of Capital (R)", 0.04, 0.20, 0.09, 0.005, format="%.3f",
                  help="Discount rate / required return. Usually WACC.")
    g = st.slider("Perpetual Growth Rate (G)", 0.00, 0.08, 0.03, 0.005, format="%.3f",
                  help="Long-term sustainable growth rate. Must be < R.")
    lifo_reserve = st.number_input("LIFO Reserve ($M, if applicable)", min_value=0.0,
                                   value=0.0, step=10.0, format="%.1f") * 1e6

    st.divider()
    st.caption("Data: Yahoo Finance · 5-min cache")
    st.caption("Formulas: Hooke Ch.1–16")
    run = st.button("🔍 Analyse", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 Value Investing Toolkit")

if g >= r:
    st.error("⚠️ Growth rate G must be strictly less than Cost of Capital R.")
    st.stop()

if not run and "last_ticker" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click **Analyse** to begin.")
    st.stop()

ticker = st.session_state.get("last_ticker", ticker_input) if not run else ticker_input
if run:
    st.session_state["last_ticker"] = ticker_input
    ticker = ticker_input

# ── Fetch ──
with st.spinner(f"Fetching data for **{ticker}**…"):
    try:
        info, inc, bal, cf, hist, ttm_div = fetch_data(ticker)
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
    st.error(f"No data found for ticker **{ticker}**. Check the symbol and try again.")
    st.stop()

d  = extract_financials(info, inc, bal, cf, ttm_div)
v  = compute_valuations(d, r, g, lifo_reserve)

# ── Total Return & Relative Strength from hist ──
if not hist.empty:
    price_start = float(hist["Close"].iloc[0])
    price_end   = float(hist["Close"].iloc[-1])
    v["total_return"] = (price_end - price_start + d["dividend_ttm"]) / price_start
    # Relative Strength: fetch SPY for same period to compare
    try:
        spy_hist = yf.Ticker("SPY").history(period="1y")
        if not spy_hist.empty:
            spy_ret = (float(spy_hist["Close"].iloc[-1]) - float(spy_hist["Close"].iloc[0])) / float(spy_hist["Close"].iloc[0])
            v["relative_strength"] = safe_div(v["total_return"], spy_ret) if spy_ret else None
    except Exception:
        pass

# ── Company header ──
col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
with col_a:
    st.subheader(f"{d['name']}  [{ticker}]")
    st.caption(f"{d['sector']}  ·  {d['industry']}")
with col_b:
    st.metric("Price", f"${d['price']:,.2f}" if d['price'] else "N/A")
with col_c:
    st.metric("Market Cap", fmt_currency(d["market_cap"]))
with col_d:
    st.metric("1Y Total Return", fmt_pct(v["total_return"]) if v["total_return"] else "N/A")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Overview",
    "💰 Valuation",
    "🚀 Growth Analysis",
    "🏭 Asset-Based",
    "🎯 Summary",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enterprise Value",  fmt_currency(v["ev"]))
    c2.metric("P/E Ratio",         fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A")
    c3.metric("P/B Ratio",         fmt_x(d["pb_ratio"]) if d["pb_ratio"] else "N/A")
    c4.metric("P/S Ratio",         fmt_x(d["ps_ratio"]) if d["ps_ratio"] else "N/A")

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Beta",             f"{d['beta']:.2f}" if d['beta'] else "N/A")
    c2b.metric("Dividend Yield",   fmt_pct(d["dividend_yield"]) if d["dividend_yield"] else "0.0%")
    c3b.metric("1Y Total Return",  fmt_pct(v["total_return"]) if v["total_return"] else "N/A")
    rs = v["relative_strength"]
    c4b.metric("Rel. Strength vs SPY", f"{rs:.2f}×" if rs else "N/A",
               help="Stock 1Y return / SPY 1Y return. >1 = outperforming market.")

    st.markdown('<div class="section-header">Price Chart (1 Year)</div>', unsafe_allow_html=True)
    if not hist.empty:
        chart_df = hist[["Close"]].rename(columns={"Close": "Price ($)"})
        st.line_chart(chart_df, use_container_width=True)

    st.markdown('<div class="section-header">Key Profitability</div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Gross Margin",     fmt_pct(d["gross_margin"]))
    pc2.metric("Operating Margin", fmt_pct(d["operating_margin"]))
    pc3.metric("Net Margin",       fmt_pct(d["net_margin"]))
    pc4.metric("ROE",              fmt_pct(d["roe"]))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Valuation
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Earnings Power Value (EPV)  — Greenwald</div>',
                unsafe_allow_html=True)
    st.caption("EPV = Adjusted Earnings ÷ R  (assumes zero growth — most conservative)")

    e1, e2, e3 = st.columns(3)
    e1.metric("EPV (Total)",     fmt_currency(v["epv_total"]))
    e2.metric("EPV per Share",   f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    e3.metric("Current Price",   f"${d['price']:,.2f}" if d["price"] else "N/A")

    st.markdown('<div class="section-header">Margin of Safety</div>', unsafe_allow_html=True)
    st.caption("MoS = Intrinsic Value − Market Price  (Hooke Ch.1 / Graham)")

    m1, m2, m3 = st.columns(3)
    m1.metric("MoS (per share)", f"${v['mos_per_share']:,.2f}" if v["mos_per_share"] else "N/A",
              delta=f"{fmt_pct(v['mos_pct'])} discount" if v["mos_pct"] else None)
    m2.metric("MoS (Total $)",   fmt_currency(v["mos_total"]))
    m3.metric("MoS %",           fmt_pct(v["mos_pct"]))

    st.markdown("**Signal:** " + mos_signal(v["mos_pct"]), unsafe_allow_html=True)
    st.info("Graham's rule of thumb: ≥33% margin of safety for adequate protection; ≥50% for strong buy.")

    st.markdown('<div class="section-header">Dividend Discount Model (DDM)  — Ch.6</div>',
                unsafe_allow_html=True)
    st.caption("V = Dividend ÷ (R − G)")

    if d["dividend_ttm"] and d["dividend_ttm"] > 0:
        d1, d2, d3 = st.columns(3)
        d1.metric("TTM Dividends / Share", f"${d['dividend_ttm']:,.4f}")
        d2.metric("DDM Value / Share",     f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A")
        d3.metric("DDM vs Price",
                  f"${(v['ddm_per_share'] or 0) - (d['price'] or 0):+,.2f}" if v["ddm_per_share"] and d["price"] else "N/A")
    else:
        st.warning("No dividends paid — DDM not applicable for this stock.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Growth Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Growth Factor F  — Ch.7 (Intel framework)</div>',
                unsafe_allow_html=True)
    st.caption("F = (ROC − G) ÷ (R − G).  If ROC = R then F = 1, growth adds zero value.")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROIC (approx.)",   fmt_pct(d["roic"]))
    g2.metric("ROE",              fmt_pct(d["roe"]))
    g3.metric("Growth Factor F",  fmt_x(v["growth_factor_F"]))
    g4.metric("Growth Mult. M",   fmt_x(v["growth_mult_M"]))

    roc_used = d["roic"] or d["roe"]
    if roc_used and v["growth_factor_F"]:
        if roc_used > r:
            st.success(f"✅ ROC ({fmt_pct(roc_used)}) > R ({fmt_pct(r)}) — growth **creates** value (F={fmt_x(v['growth_factor_F'])}).")
        elif abs(roc_used - r) < 0.01:
            st.warning(f"⚠️ ROC ≈ R — growth is value-neutral.")
        else:
            st.error(f"🔴 ROC ({fmt_pct(roc_used)}) < R ({fmt_pct(r)}) — growth **destroys** value.")
    else:
        st.info("ROIC/ROE data unavailable — cannot compute growth value.")

    st.markdown('<div class="section-header">PEG Ratio  — Ch.11 (Glenn Greenberg)</div>',
                unsafe_allow_html=True)
    st.caption("PEG = P/E ÷ Growth Rate (%).  < 1.0 = undervalued relative to growth.")

    pg1, pg2, pg3 = st.columns(3)
    pg1.metric("P/E Ratio",    fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A")
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    pg2.metric("Growth Rate Used", f"{g_pct:.1f}%")
    pg3.metric("PEG Ratio",    fmt_x(v["peg"]) if v["peg"] else "N/A")

    st.markdown("**Signal:** " + peg_signal(v["peg"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Revenue & Earnings Trends</div>',
                unsafe_allow_html=True)
    tr1, tr2, tr3 = st.columns(3)
    tr1.metric("Revenue Growth (TTM)", fmt_pct(d["revenue_growth"]))
    tr2.metric("Earnings Growth (TTM)", fmt_pct(d["earnings_growth"]))
    tr3.metric("EPS (TTM)", f"${d['eps']:,.2f}" if d["eps"] else "N/A")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Asset-Based
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Franchise Value  — Ch.5 (Greenwald)</div>',
                unsafe_allow_html=True)
    st.caption("Franchise Value = EPV − Reproduction Cost of Assets")

    fv1, fv2, fv3 = st.columns(3)
    fv1.metric("EPV (Total)",          fmt_currency(v["epv_total"]))
    fv2.metric("Reproduction Cost",    fmt_currency(v["reproduction_cost"]),
               help="Book value of equity + LIFO reserve (if any)")
    fv3.metric("Franchise Value",      fmt_currency(v["franchise_value"]))

    if v["franchise_value"] and v["epv_total"]:
        if v["franchise_value"] > 0:
            fv_pct = v["franchise_value"] / v["epv_total"]
            st.success(f"✅ Positive franchise value ({fmt_pct(fv_pct)} of EPV) — competitive moat likely exists.")
        else:
            st.warning("⚠️ EPV ≤ Reproduction Cost — limited evidence of a durable competitive advantage.")

    st.markdown('<div class="section-header">Reproduction Cost — Inventory Adj. (Ch.4)</div>',
                unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Book Value (Equity)",  fmt_currency((d["book_value"] or 0) * (d["shares"] or 0) if d["book_value"] and d["shares"] else d["total_equity"]))
    rc2.metric("LIFO Reserve Added",   fmt_currency(lifo_reserve))
    rc3.metric("Adj. Reproduction Cost", fmt_currency(v["reproduction_cost"]))

    st.markdown('<div class="section-header">Cap Rate  — Ch.10 (Mario Gabelli)</div>',
                unsafe_allow_html=True)
    st.caption("Cap Rate = Operating Cash Flow (EBITDA − Capex) ÷ EV")

    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("EBITDA − Capex",  fmt_currency(v["operating_cf_for_cap"]))
    ca2.metric("Enterprise Value", fmt_currency(v["ev"]))
    ca3.metric("Cap Rate",         fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A")
    st.markdown("**Signal:** " + cap_signal(v["cap_rate"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Adjusted P/E (Sonkin, Ch.16)</div>',
                unsafe_allow_html=True)
    st.caption("Strips net cash from market cap to reveal true operating multiple.")

    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Net Cash",          fmt_currency(v["net_cash"]),
               help="Cash − Total Debt")
    sp2.metric("Op. Market Cap",    fmt_currency((v["market_cap"] or 0) - v["net_cash"]))
    sp3.metric("Sonkin Adj. P/E",   fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Summary Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 🎯 Summary: {d['name']} [{ticker}]")
    st.caption(f"R = {fmt_pct(r)}  |  G = {fmt_pct(g)}  |  Analysis date: {pd.Timestamp.now().strftime('%d %b %Y')}")

    summary_rows = [
        # Category, Metric, Value, Signal
        ("Market",     "Price",               f"${d['price']:,.2f}" if d["price"] else "N/A", "—"),
        ("Market",     "Market Cap",           fmt_currency(v["market_cap"]), "—"),
        ("Market",     "Enterprise Value",     fmt_currency(v["ev"]), "—"),
        ("Market",     "1Y Total Return",      fmt_pct(v["total_return"]) if v["total_return"] else "N/A", "—"),
        ("Market",     "Rel. Strength vs SPY", f"{v['relative_strength']:.2f}×" if v["relative_strength"] else "N/A", "—"),
        ("Valuation",  "EPV / Share",          f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A", "—"),
        ("Valuation",  "Margin of Safety",     fmt_pct(v["mos_pct"]) if v["mos_pct"] else "N/A",
                                               "✅ BUY" if (v["mos_pct"] or 0) >= 0.33 else ("⚠️ HOLD" if (v["mos_pct"] or 0) >= 0.10 else "🔴 SELL")),
        ("Valuation",  "DDM / Share",          f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A", "—"),
        ("Valuation",  "P/E Ratio",            fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A", "—"),
        ("Growth",     "ROIC",                 fmt_pct(d["roic"]) if d["roic"] else "N/A", "—"),
        ("Growth",     "Growth Factor F",      fmt_x(v["growth_factor_F"]) if v["growth_factor_F"] else "N/A",
                                               "✅ Value" if (v["growth_factor_F"] or 0) > 1 else "⚠️ Neutral"),
        ("Growth",     "PEG Ratio",            fmt_x(v["peg"]) if v["peg"] else "N/A",
                                               "✅ <1" if v["peg"] and v["peg"] < 1 else ("⚠️ 1–3" if v["peg"] and v["peg"] < 3 else "🔴 >3")),
        ("Asset",      "Franchise Value",      fmt_currency(v["franchise_value"]),
                                               "✅ Moat" if (v["franchise_value"] or 0) > 0 else "⚠️ None"),
        ("Asset",      "Cap Rate",             fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A",
                                               "✅ >8%" if (v["cap_rate"] or 0) >= 0.08 else ("⚠️ 5–8%" if (v["cap_rate"] or 0) >= 0.05 else "🔴 <5%")),
        ("Asset",      "Sonkin Adj. P/E",      fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A", "—"),
    ]

    df_summary = pd.DataFrame(summary_rows, columns=["Category", "Metric", "Value", "Signal"])

    # Style: colour Signal column
    def style_signal(val):
        if "✅" in str(val):
            return "color: #00e676; font-weight: 600"
        if "⚠️" in str(val):
            return "color: #ffd740; font-weight: 600"
        if "🔴" in str(val):
            return "color: #ff5252; font-weight: 600"
        return "color: #888"

    styled = df_summary.style.applymap(style_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=530)

    # ── Overall verdict ──
    buy_signals  = sum(1 for _, _, _, sig in summary_rows if "✅" in sig)
    warn_signals = sum(1 for _, _, _, sig in summary_rows if "⚠️" in sig)
    sell_signals = sum(1 for _, _, _, sig in summary_rows if "🔴" in sig)

    st.markdown("---")
    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("✅ Buy Signals",    buy_signals)
    vc2.metric("⚠️ Hold Signals",   warn_signals)
    vc3.metric("🔴 Sell Signals",   sell_signals)

    if buy_signals > sell_signals and buy_signals >= 3:
        vc4.success("🟢 OVERALL: ATTRACTIVE")
    elif sell_signals >= 3:
        vc4.error("🔴 OVERALL: EXPENSIVE")
    else:
        vc4.warning("🟡 OVERALL: MIXED / REVIEW")

    st.caption("⚠️ This tool is for research only. Not financial advice. Always verify data independently.")
