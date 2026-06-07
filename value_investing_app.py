"""
Value Investing Toolkit — Streamlit App
Based on: Value Investing: From Graham to Buffett and Beyond (Greenwald, Kahn, Sonkin, van Biema)
Data: Financial Modeling Prep (FMP) API  —  financialmodelingprep.com

WHY FMP INSTEAD OF ALPHA VANTAGE?
───────────────────────────────────
  Alpha Vantage free tier: 25 calls/day
  FMP free tier:          250 calls/day  (10× more)
  FMP rate limit:         300 calls/min  (vs 5/min)

FMP ENDPOINTS USED (stable/ prefix):
  1. /profile/{symbol}             — company info, price, market cap, beta, sector
  2. /income-statement/{symbol}    — revenue, EBIT, net income, tax, D&A, EPS
  3. /balance-sheet-statement/{symbol} — debt, cash, equity
  4. /cash-flow-statement/{symbol} — capex, operating CF, dividends, D&A
  5. /key-metrics-ttm/{symbol}     — pre-calculated ROIC, EV, PE, PB, margins
  6. /historical-price-full/{symbol} — 1-year price chart

CALL BUDGET:
  Lean  (2 calls): profile + income-statement        → ~125 analyses/day
  Full  (4 calls): + balance-sheet + cash-flow       → ~62 analyses/day
  Chart (6 calls): + key-metrics-ttm + historical    → ~41 analyses/day

═══════════════════════════════════════════════════════════════════════════════
BOOK OVERVIEW — THREE SLICES OF VALUE (Ch. 3, Figure 3.1)
═══════════════════════════════════════════════════════════════════════════════
  Slice 1 — ASSET VALUE  (Reproduction Cost)  most reliable
  Slice 2 — EPV          (zero-growth)         moderately reliable
  Slice 3 — PV w/ Growth (ROC > R required)   least reliable
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Value Investing Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
  .book-note {
    background: #1a1a2e; border-left: 3px solid #ffd740;
    padding: 10px 14px; border-radius: 4px;
    font-size: 13px; color: #ccc; margin: 8px 0 14px 0;
  }
  .call-meter {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    background: #111; border: 1px solid #333; border-radius: 6px;
    padding: 8px 12px; margin: 6px 0; line-height: 1.9;
  }
  .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_currency(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.{decimals}f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.{decimals}f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.{decimals}f}M"
    return f"${v:,.{decimals}f}"

def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v*100:.{decimals}f}%"

def fmt_x(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:.{decimals}f}×"

def safe_div(a, b):
    try:
        if b is None or b == 0 or np.isnan(float(b)): return None
        return float(a) / float(b)
    except: return None

def safe_float(v, default=None):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except: return default

def signal_html(label, cls):
    return f'<span class="{cls}">{label}</span>'

def book_note(text):
    return f'<div class="book-note">📖 {text}</div>'


# ── FMP fetch layer ───────────────────────────────────────────────────────────

BASE = "https://financialmodelingprep.com/stable"

@st.cache_data(ttl=28800, show_spinner=False)   # 8-hour cache per (endpoint, ticker)
def fmp_get(endpoint, ticker, api_key, **params):
    """
    Single FMP call. Cached 8 hours per (endpoint, ticker).
    Re-running the same ticker costs zero calls within the cache window.
    FMP free tier: 250 calls/day, no per-minute limit worth worrying about.
    """
    url = f"{BASE}/{endpoint}/{ticker}"
    p = {"apikey": api_key}
    p.update(params)
    r = requests.get(url, params=p, timeout=15)
    r.raise_for_status()
    data = r.json()
    # FMP returns a list for statements, dict for errors
    if isinstance(data, dict):
        msg = data.get("message") or data.get("error") or str(data)
        raise RuntimeError(f"FMP error: {msg}")
    return data


def _cache_key(ticker, mode):
    return f"{ticker.upper()}|{mode}"


def fetch_data(ticker, api_key, mode):
    """
    Tiered fetch — only calls the endpoints the selected mode needs.
    0.25s pause between calls is a courtesy; FMP free tier has no strict
    per-second limit unlike Alpha Vantage.

    Returns dict of raw FMP responses. Missing endpoints → [] handled gracefully.
    """
    raw = {
        "profile":   [],
        "income":    [],
        "balance":   [],
        "cashflow":  [],
        "metrics":   [],
        "history":   {},
    }

    def get(endpoint, key, **kw):
        raw[key] = fmp_get(endpoint, ticker, api_key, **kw)
        time.sleep(0.25)

    # Lean — always fetch (2 calls)
    get("profile",            "profile")
    get("income-statement",   "income",  limit=2)   # 2 years for YOY growth

    # Full — adds balance sheet + cash flow (2 more calls)
    if mode in ("Full (4 calls)", "Chart (6 calls)"):
        get("balance-sheet-statement", "balance",  limit=1)
        get("cash-flow-statement",     "cashflow", limit=1)

    # Chart — adds pre-calc metrics + price history (2 more calls)
    if mode == "Chart (6 calls)":
        get("key-metrics-ttm",         "metrics")
        # historical-price-full returns {"symbol":…,"historical":[…]}
        raw["history"] = fmp_get(
            "historical-price-full", ticker, api_key,
            timeseries=252
        )
        time.sleep(0.25)

    return raw


# ── Data extraction ───────────────────────────────────────────────────────────

def first(lst):
    """Most recent period from FMP list (index 0 = most recent)."""
    return lst[0] if lst else {}

def second(lst):
    """Prior period for YOY growth."""
    return lst[1] if len(lst) > 1 else {}

def extract_financials(raw):
    """
    Map FMP responses to the clean financial dict used by compute_valuations().

    FMP field reference (stable/ endpoints):
      profile:  companyName, sector, industry, price, mktCap, beta,
                volAvg, lastDiv, currency
      income:   revenue, operatingIncome, netIncome, eps, epsDiluted,
                grossProfit, ebitda, incomeTaxExpense, incomeBeforeTax,
                depreciationAndAmortization, weightedAverageShsOut
      balance:  totalDebt, cashAndCashEquivalents,
                totalStockholdersEquity, totalEquity,
                shortTermDebt, longTermDebt
      cashflow: capitalExpenditure, operatingCashFlow,
                dividendsPaid, freeCashFlow,
                depreciationAndAmortization
      metrics:  roicTTM, returnOnEquityTTM, peRatioTTM,
                pbRatioTTM, priceToSalesRatioTTM,
                enterpriseValueTTM, evToEbitdaTTM,
                netProfitMarginTTM, operatingProfitMarginTTM,
                grossProfitMarginTTM, revenuePerShareTTM
    """
    d = {}

    prof  = first(raw["profile"])       # single-item list
    inc   = first(raw["income"])
    inc2  = second(raw["income"])
    bal   = first(raw["balance"])
    cf    = first(raw["cashflow"])
    met   = first(raw["metrics"]) if raw["metrics"] else {}

    # ── Identity ──
    d["name"]     = prof.get("companyName", "")
    d["sector"]   = prof.get("sector", "N/A")
    d["industry"] = prof.get("industry", "N/A")

    # ── Price & market (profile always available) ──
    d["price"]      = safe_float(prof.get("price"))
    d["shares"]     = safe_float(prof.get("volAvg"))   # volAvg not shares — fix below
    # FMP profile gives mktCap and price; derive shares
    d["market_cap"] = safe_float(prof.get("mktCap"))
    d["shares"]     = safe_div(d["market_cap"], d["price"]) if d["market_cap"] and d["price"] else None
    d["beta"]       = safe_float(prof.get("beta"))
    d["price_is_approx"] = False   # FMP profile always has live EOD price

    # ── Income statement ──
    d["revenue"]       = safe_float(inc.get("revenue"))
    d["revenue_prior"] = safe_float(inc2.get("revenue"))
    d["net_income"]    = safe_float(inc.get("netIncome"))
    d["gross_profit"]  = safe_float(inc.get("grossProfit"))
    d["ebitda"]        = safe_float(inc.get("ebitda"))
    d["eps"]           = safe_float(inc.get("epsDiluted")) or safe_float(inc.get("eps"))

    # EBIT = operatingIncome in FMP (Ch. 6: preferred EPV starting point)
    d["ebit"] = safe_float(inc.get("operatingIncome"))
    if not d["ebit"] and d["ebitda"] and inc.get("depreciationAndAmortization"):
        d["ebit"] = d["ebitda"] - abs(safe_float(inc.get("depreciationAndAmortization"), 0))

    # D&A  (income statement field in FMP)
    d["da"] = abs(safe_float(inc.get("depreciationAndAmortization"), 0) or 0)
    # Cash flow D&A is more reliable when available
    if cf.get("depreciationAndAmortization"):
        d["da"] = abs(safe_float(cf.get("depreciationAndAmortization"), 0) or 0)

    # Tax rate
    income_tax = safe_float(inc.get("incomeTaxExpense"), 0) or 0
    pretax     = safe_float(inc.get("incomeBeforeTax"), 0) or 0
    if pretax and pretax > 0 and income_tax:
        d["tax_rate"] = min(max(income_tax / pretax, 0.10), 0.45)
    else:
        d["tax_rate"] = 0.21

    # ── Balance sheet ──
    d["total_debt"]   = safe_float(bal.get("totalDebt")) or \
                        (safe_float(bal.get("longTermDebt"), 0) or 0) + \
                        (safe_float(bal.get("shortTermDebt"), 0) or 0)
    d["cash"]         = safe_float(bal.get("cashAndCashEquivalents")) or \
                        safe_float(bal.get("cashAndShortTermInvestments"))
    d["total_equity"] = safe_float(bal.get("totalStockholdersEquity")) or \
                        safe_float(bal.get("totalEquity"))

    # Lean mode fallbacks: derive from profile + income
    if not d["total_equity"] and d["market_cap"] and prof.get("priceToBookRatio") is None:
        # estimate equity from book value if available in metrics
        pass
    d["book_value_ps"] = safe_div(d["total_equity"], d["shares"]) \
                         if d["total_equity"] and d["shares"] else None
    d["debt_cash_estimated"] = not bool(bal)

    # ── Cash flow ──
    d["capex"]        = abs(safe_float(cf.get("capitalExpenditure"), 0) or 0)
    d["operating_cf"] = safe_float(cf.get("operatingCashFlow"), 0)
    d["dividends_paid"]= abs(safe_float(cf.get("dividendsPaid"), 0) or 0)

    # Lean-mode D&A estimate: EBITDA − EBIT
    if not d["da"] and d["ebitda"] and d["ebit"]:
        d["da"] = max(d["ebitda"] - d["ebit"], 0)
        d["da_estimated"] = True
    else:
        d["da_estimated"] = False

    # Lean-mode capex estimate: 3% of revenue
    if not d["capex"] and d["revenue"]:
        d["capex"] = d["revenue"] * 0.03
        d["capex_estimated"] = True
    else:
        d["capex_estimated"] = False

    # Dividend per share (profile has lastDiv = annual dividend)
    d["dividend_ttm"] = safe_float(prof.get("lastDiv"), 0)

    # ── Multiples — prefer key-metrics-ttm when available (Chart mode) ──
    d["pe_ratio"]      = safe_float(met.get("peRatioTTM"))    or safe_float(inc.get("pe"))
    d["pb_ratio"]      = safe_float(met.get("pbRatioTTM"))
    d["ps_ratio"]      = safe_float(met.get("priceToSalesRatioTTM"))
    d["dividend_yield"]= safe_div(d["dividend_ttm"], d["price"]) if d["price"] else 0

    # EV in dollars
    d["ev_dollars"]    = safe_float(met.get("enterpriseValueTTM")) or \
                         ((d["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0)
                          if d["market_cap"] else None)

    # ── Profitability (key-metrics-ttm if available, else from income) ──
    d["roe"]              = safe_float(met.get("returnOnEquityTTM"))
    d["operating_margin"] = safe_float(met.get("operatingProfitMarginTTM")) or \
                            safe_div(d["ebit"], d["revenue"])
    d["net_margin"]       = safe_float(met.get("netProfitMarginTTM")) or \
                            safe_div(d["net_income"], d["revenue"])
    d["gross_margin"]     = safe_float(met.get("grossProfitMarginTTM")) or \
                            safe_div(d["gross_profit"], d["revenue"])

    # ── Growth ──
    if d["revenue"] and d["revenue_prior"] and d["revenue_prior"] != 0:
        d["revenue_growth"] = (d["revenue"] - d["revenue_prior"]) / abs(d["revenue_prior"])
    else:
        d["revenue_growth"] = None
    d["earnings_growth"] = None   # will compute below if data available
    ni2 = safe_float(inc2.get("netIncome"))
    if d["net_income"] and ni2 and ni2 != 0:
        d["earnings_growth"] = (d["net_income"] - ni2) / abs(ni2)

    # ── ROIC — book-correct formula (Ch. 7) ──────────────────────────────────
    # NOPAT = EBIT × (1 − tax_rate)
    # Surplus Cash = Cash − 1% of Revenue
    # Invested Capital = Equity + Debt − Surplus Cash
    # ROIC = NOPAT / Invested Capital
    try:
        op_cash_needed  = (d["revenue"] or 0) * 0.01
        surplus_cash    = max((d["cash"] or 0) - op_cash_needed, 0)
        nopat           = (d["ebit"] or 0) * (1 - d["tax_rate"])
        inv_cap         = (d["total_equity"] or 0) + (d["total_debt"] or 0) - surplus_cash
        d["roic"]           = safe_div(nopat, inv_cap) if inv_cap and inv_cap > 0 else None
        d["surplus_cash"]   = surplus_cash
        d["nopat"]          = nopat
        d["invested_capital"] = inv_cap
    except:
        d["roic"] = None
        d["surplus_cash"] = d["nopat"] = d["invested_capital"] = 0

    # Also use FMP's pre-calculated ROIC from key-metrics if available & ROIC is None
    if not d["roic"] and met.get("roicTTM"):
        d["roic"] = safe_float(met.get("roicTTM"))

    # ── 1-year price history (Chart mode only) ──
    hist = raw["history"].get("historical", [])
    if hist:
        hist_sorted = sorted(hist, key=lambda x: x["date"])[-252:]
        prices = [safe_float(h["close"]) for h in hist_sorted]
        dates  = [h["date"] for h in hist_sorted]
        d["price_history"] = pd.Series(prices, index=pd.to_datetime(dates), name="Price ($)")
    else:
        d["price_history"] = None

    return d


# ── Valuations ────────────────────────────────────────────────────────────────

def compute_valuations(d, r, g, lifo_reserve=0.0):
    v = {}
    rg = r - g

    v["market_cap"] = d["market_cap"] or (
        (d["price"] or 0) * (d["shares"] or 0) if d["price"] and d["shares"] else None
    )
    v["ev"] = d["ev_dollars"] or (
        (v["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0)
    )

    # ── Asset Value / Reproduction Cost (Ch. 4) ──────────────────────────────
    book_total = (
        (d["book_value_ps"] or 0) * (d["shares"] or 0)
        if d.get("book_value_ps") and d["shares"]
        else (d["total_equity"] or 0)
    )
    v["reproduction_cost"] = book_total + lifo_reserve

    # ── EPV (Ch. 3, 5, 6, 7) ─────────────────────────────────────────────────
    # NOPAT + 25% D&A addback (Ch. 7 conservative maintenance capex buffer)
    # ÷ R  → enterprise EPV  → subtract debt, add surplus cash → equity EPV
    if d["ebit"]:
        nopat_epv  = d["ebit"] * (1 - d["tax_rate"])
        da_addback = (d["da"] or 0) * 0.25
        adj_earn   = nopat_epv + da_addback
    else:
        adj_earn   = d["net_income"] or ((d["eps"] or 0) * (d["shares"] or 1))

    epv_enterprise      = safe_div(adj_earn, r)
    surplus             = d.get("surplus_cash") or 0
    v["epv_total"]      = (epv_enterprise - (d["total_debt"] or 0) + surplus) \
                          if epv_enterprise is not None else None
    v["epv_per_share"]  = safe_div(v["epv_total"], d["shares"])
    v["adj_earnings"]   = adj_earn

    # ── Margin of Safety ──────────────────────────────────────────────────────
    v["mos_per_share"] = (
        (v["epv_per_share"] or 0) - (d["price"] or 0)
        if v["epv_per_share"] and d["price"] else None
    )
    v["mos_total"]  = (v["epv_total"] or 0) - (v["market_cap"] or 0) \
                      if v["epv_total"] else None
    v["mos_pct"]    = safe_div(v["mos_per_share"], v["epv_per_share"]) \
                      if v["epv_per_share"] and v["epv_per_share"] != 0 else None

    # ── Franchise Value = EPV − Reproduction Cost (Ch. 5) ────────────────────
    v["franchise_value"]     = (v["epv_total"] or 0) - v["reproduction_cost"] \
                               if v["epv_total"] else None
    v["franchise_per_share"] = safe_div(v["franchise_value"], d["shares"])

    # ── PV with Growth = C × (ROC − G) / (R − G)  (Ch. 7 Appendix) ──────────
    roc = d["roic"] or d["roe"] or 0
    v["growth_factor_F"] = safe_div(roc - g, rg) if rg > 0 and roc else None

    if rg > 0 and roc and d.get("invested_capital") and (d["invested_capital"] or 0) > 0:
        pv_ent              = d["invested_capital"] * safe_div(roc - g, rg)
        v["pv_with_growth"] = (pv_ent - (d["total_debt"] or 0) + surplus) if pv_ent else None
        v["pv_per_share"]   = safe_div(v["pv_with_growth"], d["shares"])
    else:
        v["pv_with_growth"] = v["pv_per_share"] = None

    # Growth Multiplier M = [1 − (G/R)(R/ROC)] / [1 − (G/R)]  (Ch. 7)
    if r > 0 and roc and roc > 0:
        gr    = g / r
        denom = 1 - gr * (r / roc)
        v["growth_mult_M"] = safe_div(1 - gr, denom) if denom != 0 else None
    else:
        v["growth_mult_M"] = None

    # ── DDM  V = D / (R − G) (Ch. 6) ─────────────────────────────────────────
    if rg > 0 and d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        v["ddm_per_share"] = d["dividend_ttm"] / rg
        v["ddm_total"]     = v["ddm_per_share"] * (d["shares"] or 0)
    else:
        v["ddm_per_share"] = v["ddm_total"] = None

    # ── Cap Rate  (EBITDA − Capex) / EV  (Ch. 10 — Gabelli) ─────────────────
    op_cf = (d["ebitda"] or 0) - (d["capex"] or 0)
    v["cap_rate"]             = safe_div(op_cf, v["ev"]) if v["ev"] else None
    v["operating_cf_for_cap"] = op_cf

    # ── PEG  (Ch. 11 — Greenberg) ─────────────────────────────────────────────
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    v["peg"] = safe_div(d["pe_ratio"], g_pct) if d["pe_ratio"] and g_pct else None

    # ── Sonkin Adjusted P/E  (Ch. 16) ─────────────────────────────────────────
    net_cash    = (d["cash"] or 0) - (d["total_debt"] or 0)
    int_on_cash = net_cash * 0.04 if net_cash > 0 else 0
    op_mktcap   = (v["market_cap"] or 0) - net_cash
    op_earn     = (d["net_income"] or 0) - int_on_cash
    v["sonkin_pe"] = safe_div(op_mktcap, op_earn) if op_earn and op_mktcap > 0 else None
    v["net_cash"]  = net_cash

    # ── 1Y total return ───────────────────────────────────────────────────────
    v["total_return"] = None
    if d["price_history"] is not None and len(d["price_history"]) > 1:
        p0 = d["price_history"].iloc[0]
        p1 = d["price_history"].iloc[-1]
        v["total_return"] = safe_div(p1 - p0 + (d["dividend_ttm"] or 0), p0)

    return v


# ── Signals ───────────────────────────────────────────────────────────────────

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
    st.caption("Greenwald · Graham · Hooke methodology")
    st.divider()

    api_key = st.text_input(
        "FMP API Key", type="password",
        help="Free key at financialmodelingprep.com/register — 250 calls/day"
    )
    if not api_key:
        st.info("Get a free key at [financialmodelingprep.com](https://financialmodelingprep.com/register)")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("#### 📡 Data Mode")
    mode = st.radio(
        "API calls per analysis:",
        options=["Lean (2 calls)", "Full (4 calls)", "Chart (6 calls)"],
        index=1,
        help=(
            "Lean: profile + income — covers most metrics, D&A/debt estimated.\n"
            "Full: + balance sheet + cash flow — precise debt, cash, capex, D&A.\n"
            "Chart: + key-metrics-ttm + price history — pre-calc ratios + chart."
        )
    )

    calls_this     = {"Lean (2 calls)": 2, "Full (4 calls)": 4, "Chart (6 calls)": 6}[mode]
    calls_used     = st.session_state.get("calls_used_session", 0)
    cached_tickers = st.session_state.get("cached_tickers", set())
    is_cached      = _cache_key(ticker_input, mode) in cached_tickers
    budget_left    = max(250 - calls_used, 0)
    analyses_left  = budget_left // calls_this if calls_this else 0
    cache_note     = "⚡ Cached — 0 calls" if is_cached else f"📡 Fresh — {calls_this} calls"

    st.markdown(
        f'<div class="call-meter">'
        f'🔋 Session calls used: <b>{calls_used}</b><br/>'
        f'📊 Next run: <b>{cache_note}</b><br/>'
        f'🔄 ~<b>{analyses_left}</b> fresh analyses left<br/>'
        f'<small style="color:#666">FMP free tier: 250 calls/day</small>'
        f'</div>',
        unsafe_allow_html=True
    )
    if budget_left <= 20 and not is_cached:
        st.warning("⚠️ Running low — switch to Lean or re-run cached tickers.")

    st.markdown("#### Assumptions")
    r = st.slider("Cost of Capital (R)", 0.04, 0.20, 0.09, 0.005, format="%.3f",
                  help="Ch. 7: '12% is reasonable — several points above long-term S&P 500 return.'")
    g = st.slider("Perpetual Growth Rate (G)", 0.00, 0.08, 0.03, 0.005, format="%.3f",
                  help="Ch. 7: 'If G ≥ R, value becomes infinite — R is the realistic ceiling.'")
    lifo_reserve = st.number_input(
        "LIFO Reserve ($M)", min_value=0.0, value=0.0, step=10.0, format="%.1f",
        help="Ch. 4: From 10-K footnotes. Converts LIFO inventory to replacement cost."
    ) * 1e6

    st.divider()
    run = st.button("🔍 Analyse", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 Value Investing Toolkit")
st.caption("Based on *Value Investing: From Graham to Buffett and Beyond* (Greenwald et al.)  ·  Data: Financial Modeling Prep")

if g >= r:
    st.error("⚠️ G must be < R. (Ch. 7: if G ≥ R, value becomes infinite.)")
    st.stop()
if not api_key:
    st.warning("Enter your FMP API key in the sidebar. Free at financialmodelingprep.com/register")
    st.stop()
if not run and "last_ticker" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click **Analyse**.")
    st.stop()

ticker   = ticker_input if run else st.session_state.get("last_ticker", ticker_input)
if run:
    st.session_state["last_ticker"] = ticker_input
    st.session_state["last_mode"]   = mode
    ck    = _cache_key(ticker_input, mode)
    known = st.session_state.get("cached_tickers", set())
    if ck not in known:
        st.session_state["calls_used_session"] = \
            st.session_state.get("calls_used_session", 0) + calls_this
        known.add(ck)
        st.session_state["cached_tickers"] = known

mode_used = mode if run else st.session_state.get("last_mode", mode)

with st.spinner(f"Fetching {ticker} via FMP ({mode_used}) …"):
    try:
        raw = fetch_data(ticker, api_key, mode_used)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Network error: {e}")
        st.stop()

if not raw["profile"]:
    st.error(f"No data for **{ticker}**. Check the ticker symbol and your FMP API key.")
    st.stop()

d = extract_financials(raw)
v = compute_valuations(d, r, g, lifo_reserve)

# ── Data quality notes ────────────────────────────────────────────────────────
lean = mode_used == "Lean (2 calls)"
notes = []
if lean:
    notes.append("**Lean mode:** balance sheet and cash flow not fetched — debt/cash/capex estimated.")
if d.get("debt_cash_estimated"):
    notes.append("Debt & cash unavailable in Lean mode — EPV debt adjustment skipped.")
if d.get("da_estimated"):
    notes.append("D&A estimated as EBITDA − EBIT (Full mode gives the exact figure).")
if d.get("capex_estimated"):
    notes.append("Capex estimated at 3% of revenue (Full mode gives the exact figure).")
if notes:
    with st.expander(f"⚠️ {len(notes)} data note(s)", expanded=lean):
        for n in notes:
            st.caption(f"• {n}")

# ── Company header ─────────────────────────────────────────────────────────────
ca, cb, cc, cd = st.columns([3, 1, 1, 1])
with ca:
    st.subheader(f"{d['name']}  [{ticker}]")
    st.caption(f"{d['sector']}  ·  {d['industry']}  ·  {mode_used}")
cb.metric("Price",      f"${d['price']:,.2f}" if d["price"] else "N/A")
cc.metric("Market Cap", fmt_currency(v["market_cap"]))
cd.metric("1Y Return",  fmt_pct(v["total_return"]) if v["total_return"] else "N/A")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Overview", "💰 Valuation (EPV)", "🚀 Growth Analysis", "🏭 Asset & Franchise", "🎯 Summary"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(book_note(
        "Ch. 2: Market price is not intrinsic value — it may be wrong. "
        "The value investor determines intrinsic value independently, then compares to price."
    ), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enterprise Value",  fmt_currency(v["ev"]))
    c2.metric("P/E Ratio",         fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A")
    c3.metric("P/B Ratio",         fmt_x(d["pb_ratio"]) if d["pb_ratio"] else "N/A",
              help="P/B > 1 means market values the franchise above reproduction cost.")
    c4.metric("P/S Ratio",         fmt_x(d["ps_ratio"]) if d["ps_ratio"] else "N/A")

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Beta",            f"{d['beta']:.2f}" if d["beta"] else "N/A")
    c2b.metric("Dividend Yield",  fmt_pct(d["dividend_yield"]) if d["dividend_yield"] else "0.0%")
    c3b.metric("1Y Total Return", fmt_pct(v["total_return"]) if v["total_return"] else "N/A")
    c4b.metric("EPS (Diluted)",   f"${d['eps']:,.2f}" if d["eps"] else "N/A")

    st.markdown('<div class="section-header">Price Chart (1 Year)</div>', unsafe_allow_html=True)
    if d["price_history"] is not None:
        st.line_chart(d["price_history"], use_container_width=True)
    else:
        st.info("📡 Price chart requires **Chart (6 calls)** mode.")

    st.markdown('<div class="section-header">Profitability</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 6: Stable, high operating margins over multiple years signal a franchise. "
        "WD-40's consistent 24–27% EBIT margin was the key tell. Look for stability, not just level."
    ), unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Gross Margin",     fmt_pct(d["gross_margin"]))
    pc2.metric("Operating Margin", fmt_pct(d["operating_margin"]),
               help="EBIT ÷ Revenue — the book's preferred profitability measure")
    pc3.metric("Net Margin",       fmt_pct(d["net_margin"]))
    pc4.metric("ROE",              fmt_pct(d["roe"]))

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Earnings Power Value (EPV) — Greenwald</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 3: EPV = Adjusted Earnings ÷ R. Zero-growth, most conservative. "
        "Ch. 6/7: Start with EBIT, apply tax → NOPAT, add back 25% D&A (maintenance capex buffer), "
        "subtract debt, add surplus cash → equity EPV."
    ), unsafe_allow_html=True)

    with st.expander("📐 EPV Build-up", expanded=True):
        eb1, eb2, eb3 = st.columns(3)
        eb1.metric("EBIT (Operating Income)", fmt_currency(d["ebit"]))
        eb2.metric("Tax Rate",                fmt_pct(d["tax_rate"]))
        eb3.metric("NOPAT = EBIT × (1−tax)",  fmt_currency(d.get("nopat")))
        eb4, eb5, eb6 = st.columns(3)
        da_add = (d["da"] or 0) * 0.25
        eb4.metric("D&A Addback (25%)" + (" est." if d.get("da_estimated") else ""),
                   fmt_currency(da_add),
                   help="Ch. 7 Intel: add back 25% of D&A as conservative maintenance capex buffer")
        eb5.metric("Surplus Cash Added", fmt_currency(d.get("surplus_cash")),
                   help="Ch. 7: cash > 1% of sales. Added to equity EPV.")
        eb6.metric("Debt Subtracted",    fmt_currency(d.get("total_debt")),
                   help="Ch. 7: subtract debt to convert enterprise EPV to equity EPV.")
        st.caption("EPV (equity) = (NOPAT + 25% D&A) ÷ R  −  Debt  +  Surplus Cash")

    e1, e2, e3 = st.columns(3)
    e1.metric("EPV (Total Equity)", fmt_currency(v["epv_total"]))
    e2.metric("EPV per Share",      f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    e3.metric("Current Price",      f"${d['price']:,.2f}" if d["price"] else "N/A")

    st.markdown('<div class="section-header">Margin of Safety</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Graham / Ch. 3: MoS = (EPV − Price) / EPV. "
        "≥33% = adequate protection (BUY). ≥50% = strong buy. "
        "Negative MoS = market pricing in growth — only justified with a confirmed franchise."
    ), unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("MoS per Share", f"${v['mos_per_share']:,.2f}" if v["mos_per_share"] else "N/A",
              delta=fmt_pct(v["mos_pct"]) if v["mos_pct"] else None)
    m2.metric("MoS Total",     fmt_currency(v["mos_total"]))
    m3.metric("MoS %",         fmt_pct(v["mos_pct"]))
    st.markdown("**Signal:** " + mos_signal(v["mos_pct"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dividend Discount Model (DDM) — Ch. 6</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "V = Dividend ÷ (R − G). Cross-check for dividend-paying franchises (Ch. 6 WD-40). "
        "Only reliable when dividends represent stable distributable cash flow."
    ), unsafe_allow_html=True)
    if d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        d1, d2, d3 = st.columns(3)
        d1.metric("Dividend / Share", f"${d['dividend_ttm']:,.4f}")
        d2.metric("DDM Value",        f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A")
        d3.metric("DDM vs Price",
                  f"${(v['ddm_per_share'] or 0) - (d['price'] or 0):+,.2f}"
                  if v["ddm_per_share"] and d["price"] else "N/A")
    else:
        st.warning("No dividends — DDM not applicable.")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Growth Factor F & PV with Growth — Ch. 7</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7: PV = C × (ROC − G) / (R − G)  where C = Invested Capital. "
        "F = (ROC − G) / (R − G). F < 1 = growth destroys value. F = 1 = neutral. F > 1 = creates value. "
        "'Growth only adds value when ROC > R.' — Ch. 7"
    ), unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROIC",            fmt_pct(d["roic"]),          help="NOPAT ÷ Invested Capital (Ch. 7 formula)")
    g2.metric("ROE",             fmt_pct(d["roe"]),            help="Fallback when ROIC unavailable")
    g3.metric("Growth Factor F", fmt_x(v["growth_factor_F"]), help="F = (ROC−G)/(R−G). F>1 = value-creating growth")
    g4.metric("Growth Mult. M",  fmt_x(v["growth_mult_M"]),   help="M = PV/EPV. How much growth adds above EPV.")

    roc_used = d["roic"] or d["roe"]
    if roc_used and v["growth_factor_F"]:
        if roc_used > r:
            st.success(f"✅ ROC ({fmt_pct(roc_used)}) > R ({fmt_pct(r)}) — growth creates value (F = {fmt_x(v['growth_factor_F'])}).")
        elif abs(roc_used - r) < 0.01:
            st.warning("⚠️ ROC ≈ R — growth is value-neutral. Don't pay a premium for it.")
        else:
            st.error(f"🔴 ROC ({fmt_pct(roc_used)}) < R ({fmt_pct(r)}) — growth destroys value (F = {fmt_x(v['growth_factor_F'])}).")

    st.markdown('<div class="section-header">PV with Growth vs EPV</div>', unsafe_allow_html=True)
    pv1, pv2, pv3 = st.columns(3)
    pv1.metric("EPV / Share (zero growth)", f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    pv2.metric("PV / Share (with growth)",  f"${v['pv_per_share']:,.2f}"  if v["pv_per_share"]  else "N/A",
               help="C × (ROC−G)/(R−G) equity-adjusted. Ch. 7 exact formula.")
    pv3.metric("Growth Premium",
               fmt_currency((v["pv_with_growth"] or 0) - (v["epv_total"] or 0))
               if v["pv_with_growth"] and v["epv_total"] else "N/A")

    st.markdown('<div class="section-header">Growth Value Matrix — Book Table 7.11</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7: PV/EPV ratios. It takes BOTH high ROC/R AND high G/R to generate meaningful "
        "growth value. Most businesses fall in the 1.0–1.5× range."
    ), unsafe_allow_html=True)
    matrix_df = pd.DataFrame({
        "G/R":       ["25%", "50%", "75%"],
        "ROC/R=1.0": [1.00,  1.00,  1.00],
        "ROC/R=1.5": [1.11,  1.33,  2.00],
        "ROC/R=2.0": [1.17,  1.50,  2.50],
        "ROC/R=2.5": [1.20,  1.60,  2.80],
        "ROC/R=3.0": [1.22,  1.67,  3.00],
    }).set_index("G/R")
    if roc_used and r > 0:
        st.caption(f"This company: ROC/R = {roc_used/r:.2f}×, G/R = {g/r:.0%} → M ≈ {fmt_x(v['growth_mult_M'])}")
    st.dataframe(matrix_df, use_container_width=True)

    st.markdown('<div class="section-header">PEG Ratio — Ch. 11 (Greenberg)</div>', unsafe_allow_html=True)
    pg1, pg2, pg3 = st.columns(3)
    pg1.metric("P/E Ratio",        fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A")
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    pg2.metric("Growth Rate Used", f"{g_pct:.1f}%")
    pg3.metric("PEG Ratio",        fmt_x(v["peg"]) if v["peg"] else "N/A")
    st.markdown("**Signal:** " + peg_signal(v["peg"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Revenue & Earnings Trends</div>', unsafe_allow_html=True)
    tr1, tr2, tr3 = st.columns(3)
    tr1.metric("Revenue Growth (YOY)",  fmt_pct(d["revenue_growth"]))
    tr2.metric("Earnings Growth (YOY)", fmt_pct(d["earnings_growth"]))
    tr3.metric("Revenue (Annual)",      fmt_currency(d["revenue"]))

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Three-Slice Valuation Summary — Ch. 3, Fig. 3.1</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Safety hierarchy (most → least reliable): Asset Value → EPV → PV with Growth. "
        "Ideal buy: Price < EPV. If EPV > Asset Value → franchise confirmed. "
        "If Price > EPV → market pricing in growth (only justified with confirmed moat)."
    ), unsafe_allow_html=True)
    sv1, sv2, sv3, sv4 = st.columns(4)
    sv1.metric("① Asset Value",     fmt_currency(v["reproduction_cost"]), help="Book equity + LIFO reserve. Ch. 4.")
    sv2.metric("② EPV (Equity)",    fmt_currency(v["epv_total"]),         help="NOPAT/R debt-adjusted. Ch. 3, 5.")
    sv3.metric("③ PV with Growth",  fmt_currency(v["pv_with_growth"]) if v["pv_with_growth"] else "N/A",
               help="C × (ROC−G)/(R−G). Only valid if franchise confirmed. Ch. 7.")
    sv4.metric("Market Cap",        fmt_currency(v["market_cap"]))

    if v["epv_total"] and v["reproduction_cost"]:
        ratio = safe_div(v["epv_total"], v["reproduction_cost"])
        if ratio and ratio > 1:
            st.success(f"✅ EPV ({fmt_currency(v['epv_total'])}) > Reproduction Cost ({fmt_currency(v['reproduction_cost'])}) by {fmt_x(ratio)} — franchise confirmed.")
        elif ratio and ratio < 0.8:
            st.error("🔴 EPV < Reproduction Cost — poor management or industry decline (Ch. 3).")
        else:
            st.warning("⚠️ EPV ≈ Reproduction Cost — no clear franchise.")

    st.markdown('<div class="section-header">Franchise Value — Ch. 5</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 5: Franchise Value = EPV − Reproduction Cost. "
        "Positive → earns above competitive return → moat. "
        "'Without barriers to entry, competition forces returns to the cost of capital.' — Ch. 5"
    ), unsafe_allow_html=True)
    fv1, fv2, fv3 = st.columns(3)
    fv1.metric("EPV (Equity)",       fmt_currency(v["epv_total"]))
    fv2.metric("Reproduction Cost",  fmt_currency(v["reproduction_cost"]))
    fv3.metric("Franchise Value",    fmt_currency(v["franchise_value"]))
    if v["franchise_value"] and v["epv_total"]:
        fv_pct = safe_div(v["franchise_value"], v["epv_total"])
        if (v["franchise_value"] or 0) > 0:
            st.success(f"✅ Franchise value = {fmt_currency(v['franchise_value'])} ({fmt_pct(fv_pct)} of EPV).")
        else:
            st.warning("⚠️ No franchise value — EPV ≤ Reproduction Cost.")

    st.markdown('<div class="section-header">ROIC Components — Ch. 7</div>', unsafe_allow_html=True)
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("NOPAT",            fmt_currency(d.get("nopat")))
    rc2.metric("Invested Capital", fmt_currency(d.get("invested_capital")))
    rc3.metric("Surplus Cash",     fmt_currency(d.get("surplus_cash")))
    rc4.metric("ROIC",             fmt_pct(d["roic"]))
    if lean:
        st.caption("⚠️ Invested capital uses estimated debt/cash in Lean mode.")

    st.markdown('<div class="section-header">Cap Rate — Ch. 10 (Gabelli)</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 10: Cap Rate = (EBITDA − Capex) ÷ EV. "
        "Gabelli's Private Market Value. Cap rate > 8% = attractively priced."
        + (" Capex estimated at 3% revenue in Lean mode." if d.get("capex_estimated") else "")
    ), unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("EBITDA − Capex",  fmt_currency(v["operating_cf_for_cap"]))
    ca2.metric("Enterprise Value",fmt_currency(v["ev"]))
    ca3.metric("Cap Rate",        fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A")
    st.markdown("**Signal:** " + cap_signal(v["cap_rate"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sonkin Adjusted P/E — Ch. 16</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 16: Strip net cash from market cap and earnings to reveal the true operating multiple. "
        "Adj. P/E = (Market Cap − Net Cash) ÷ (Net Income − Interest on Surplus Cash)."
    ), unsafe_allow_html=True)
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Net Cash",        fmt_currency(v["net_cash"]))
    sp2.metric("Op. Market Cap",  fmt_currency((v["market_cap"] or 0) - v["net_cash"]))
    sp3.metric("Sonkin Adj. P/E", fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 🎯 Summary: {d['name']} [{ticker}]")
    st.caption(f"R = {fmt_pct(r)}  |  G = {fmt_pct(g)}  |  Mode: {mode_used}  |  {pd.Timestamp.now().strftime('%d %b %Y')}")

    summary_rows = [
        ("Market",    "Price",                  f"${d['price']:,.2f}" if d["price"] else "N/A", "—"),
        ("Market",    "Market Cap",              fmt_currency(v["market_cap"]),    "—"),
        ("Market",    "Enterprise Value",        fmt_currency(v["ev"]),            "—"),
        ("Market",    "1Y Total Return",         fmt_pct(v["total_return"]) if v["total_return"] else "N/A", "—"),
        ("Valuation", "① Reproduction Cost",    fmt_currency(v["reproduction_cost"]), "—"),
        ("Valuation", "② EPV / Share",           f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A", "—"),
        ("Valuation", "③ PV w/ Growth / Share",  f"${v['pv_per_share']:,.2f}"  if v["pv_per_share"]  else "N/A", "—"),
        ("Valuation", "Margin of Safety",        fmt_pct(v["mos_pct"]) if v["mos_pct"] else "N/A",
                                                 "✅ BUY"  if (v["mos_pct"] or 0) >= 0.33
                                                 else ("⚠️ HOLD" if (v["mos_pct"] or 0) >= 0.10 else "🔴 SELL")),
        ("Valuation", "DDM / Share",             f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A", "—"),
        ("Valuation", "P/E Ratio",               fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A", "—"),
        ("Growth",    "ROIC",                    fmt_pct(d["roic"]) if d["roic"] else "N/A",
                                                 "✅ ROC>R" if (d["roic"] or 0) > r
                                                 else ("⚠️ ROC≈R" if abs((d["roic"] or 0) - r) < 0.02 else "🔴 ROC<R")),
        ("Growth",    "Growth Factor F",         fmt_x(v["growth_factor_F"]) if v["growth_factor_F"] else "N/A",
                                                 "✅ Value" if (v["growth_factor_F"] or 0) > 1 else "⚠️ Neutral"),
        ("Growth",    "PEG Ratio",               fmt_x(v["peg"]) if v["peg"] else "N/A",
                                                 "✅ <1" if v["peg"] and v["peg"] < 1
                                                 else ("⚠️ 1–3" if v["peg"] and v["peg"] < 3 else "🔴 >3")),
        ("Asset",     "Franchise Value",         fmt_currency(v["franchise_value"]),
                                                 "✅ Moat" if (v["franchise_value"] or 0) > 0 else "⚠️ None"),
        ("Asset",     "Cap Rate",                fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A",
                                                 "✅ >8%" if (v["cap_rate"] or 0) >= 0.08
                                                 else ("⚠️ 5–8%" if (v["cap_rate"] or 0) >= 0.05 else "🔴 <5%")),
        ("Asset",     "Sonkin Adj. P/E",         fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A", "—"),
    ]

    df_summary = pd.DataFrame(summary_rows, columns=["Category", "Metric", "Value", "Signal"])

    def style_signal(val):
        if "✅" in str(val): return "color: #00e676; font-weight: 600"
        if "⚠️" in str(val): return "color: #ffd740; font-weight: 600"
        if "🔴" in str(val): return "color: #ff5252; font-weight: 600"
        return "color: #888"

    st.dataframe(df_summary.style.map(style_signal, subset=["Signal"]),
                 use_container_width=True, hide_index=True, height=610)

    st.divider()
    buy_s  = sum(1 for *_, sig in summary_rows if "✅" in sig)
    warn_s = sum(1 for *_, sig in summary_rows if "⚠️" in sig)
    sell_s = sum(1 for *_, sig in summary_rows if "🔴" in sig)
    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("✅ Buy Signals",  buy_s)
    vc2.metric("⚠️ Hold Signals", warn_s)
    vc3.metric("🔴 Sell Signals", sell_s)
    if   buy_s > sell_s and buy_s >= 3: vc4.success("🟢 OVERALL: ATTRACTIVE")
    elif sell_s >= 3:                   vc4.error("🔴 OVERALL: EXPENSIVE")
    else:                               vc4.warning("🟡 OVERALL: MIXED / REVIEW")

    st.markdown(book_note(
        "Book safety hierarchy: Asset Value → EPV → PV with Growth. "
        "Ideal buy: Price < EPV < PV with Growth. "
        "'The greater the margin of safety, the lower the risk.' — Ch. 3"
    ), unsafe_allow_html=True)
    st.caption("⚠️ Research only — not financial advice. Verify all data independently.")
