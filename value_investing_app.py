"""
Value Investing Toolkit — Streamlit App
Based on: Value Investing: From Graham to Buffett and Beyond (Greenwald, Kahn, Sonkin, van Biema)
Data: Alpha Vantage API (free tier)

API CALL BUDGET
───────────────
Free tier: 25 calls/day, max 5/min.
This app uses a tiered fetch strategy to minimise call usage:

  MODE 1 — "Lean" (2 calls):  OVERVIEW + INCOME_STATEMENT
    Covers ~90% of all displayed metrics. Recommended for daily use.
    Cash / debt from OVERVIEW BookValue + assumptions; no price chart.

  MODE 2 — "Full" (4 calls):  + BALANCE_SHEET + CASH_FLOW
    Adds precise debt, cash, capex, D&A, dividends.
    Improves EPV (D&A addback), ROIC (invested capital), cap rate, DDM.

  MODE 3 — "Chart" (6 calls): + GLOBAL_QUOTE + TIME_SERIES_DAILY
    Adds live price tick and 1-year price chart.
    Only use when you need the chart or most-current intraday price.

The user selects mode in the sidebar. Each mode clearly states call count
and what improves. The app displays a running call-budget meter.

═══════════════════════════════════════════════════════════════════════════════
BOOK OVERVIEW — THREE SLICES OF VALUE (Ch. 3, Figure 3.1)
═══════════════════════════════════════════════════════════════════════════════
  Slice 1 — ASSET VALUE  (Reproduction Cost):  most reliable
  Slice 2 — EPV          (zero-growth):         moderately reliable
  Slice 3 — PV w/ Growth (ROC > R required):    least reliable
Safety hierarchy: Asset Value < EPV < PV with Growth
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
    padding: 8px 12px; margin: 6px 0;
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
    except Exception: return None

def safe_float(v, default=None):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except Exception: return default

def signal_html(label, cls):
    return f'<span class="{cls}">{label}</span>'

def book_note(text):
    return f'<div class="book-note">📖 {text}</div>'


# ── Alpha Vantage fetch ───────────────────────────────────────────────────────

BASE = "https://www.alphavantage.co/query"

# ── Per-endpoint cache (keyed by ticker + function, TTL = 8 hours) ────────────
# WHY NOT cache_data.clear() on every run?
#   Clearing the entire cache on every Analyse click forces fresh API calls
#   even for tickers already fetched minutes ago — burning your daily budget.
#   Instead we cache each (function, ticker) pair for 8 hours.
#   A re-run of the same ticker within 8 hours costs ZERO additional calls.
#   Only switching to a new ticker or a forced refresh hits the API.
@st.cache_data(ttl=28800, show_spinner=False)   # 8 hours
def av_get(function, ticker, api_key, **kwargs):
    """
    Single AV endpoint call. Cached per (function, ticker) for 8 hours.
    Free tier: 25 calls/day. Errors trigger clear user messages.
    """
    params = {"function": function, "symbol": ticker, "apikey": api_key}
    params.update(kwargs)
    r = requests.get(BASE, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "Note" in data:
        raise RuntimeError(
            "⏳ Rate limit hit (5 calls/min). Wait 60 seconds and try again."
        )
    if "Information" in data:
        raise RuntimeError(
            "🔑 Daily limit reached (25 calls/day on free tier). "
            "Try again tomorrow or upgrade at alphavantage.co/premium/"
        )
    if "Error Message" in data:
        raise RuntimeError(f"❌ Ticker '{ticker}' not found. Check the symbol.")
    return data


def _cache_key(ticker, mode):
    """Unique string identifying a (ticker, mode) fetch — used to track what's cached."""
    return f"{ticker.upper()}|{mode}"


def fetch_data(ticker, api_key, mode):
    """
    Tiered fetch — only calls the endpoints required for the selected mode.
    Each endpoint is individually cached, so changing only assumptions (R, G)
    or switching tabs NEVER triggers new API calls.

    Cache hit check:
      Before making any network call, we check whether this (ticker, mode)
      was already fetched in this session. If yes, av_get() returns from
      its own cache — zero real calls. The budget counter is only incremented
      on a genuine first fetch.

    Returns a dict of raw AV responses keyed by endpoint name.
    Missing endpoints return {} — all downstream code handles gracefully.
    """
    raw = {
        "overview": {}, "income": {}, "balance": {},
        "cashflow": {}, "quote": {}, "daily": {}
    }

    def get_with_pause(fn, key, **kw):
        raw[key] = av_get(fn, ticker, api_key, **kw)
        time.sleep(0.5)   # respect 5 calls/min burst limit

    # Core — always fetch (2 calls)
    get_with_pause("OVERVIEW",         "overview")
    get_with_pause("INCOME_STATEMENT", "income")

    # Full mode — balance sheet + cash flow (+2 calls)
    if mode in ("Full (4 calls)", "Chart (6 calls)"):
        get_with_pause("BALANCE_SHEET", "balance")
        get_with_pause("CASH_FLOW",     "cashflow")

    # Chart mode — live quote + daily prices (+2 calls)
    if mode == "Chart (6 calls)":
        get_with_pause("GLOBAL_QUOTE",      "quote")
        get_with_pause("TIME_SERIES_DAILY", "daily", outputsize="compact")

    return raw


# ── Data extraction ───────────────────────────────────────────────────────────

def latest(report_list):
    try: return report_list[0] if report_list else {}
    except: return {}

def prior(report_list):
    try: return report_list[1] if len(report_list) > 1 else {}
    except: return {}

def extract_financials(raw):
    """
    Map AV responses → clean financial dict.

    Fallback priority for each field (most-accurate to least):
      1. Dedicated statement endpoint (INCOME / BALANCE / CASHFLOW)
      2. OVERVIEW pre-computed field (TTM figures, ratios)
      3. Derived / estimated value
      4. None  (displayed as N/A, downstream calculations skip gracefully)

    This means Lean mode (no balance/cashflow) still populates most fields
    from OVERVIEW and INCOME, just with slightly lower precision.
    """
    d = {}
    ov  = raw["overview"]
    inc = latest(raw["income"].get("annualReports", []))
    inc_prior = prior(raw["income"].get("annualReports", []))
    bal = latest(raw["balance"].get("annualReports", []))   # {} in Lean mode
    cf  = latest(raw["cashflow"].get("annualReports", []))  # {} in Lean mode
    q   = raw["quote"].get("Global Quote", {})              # {} unless Chart mode

    # ── Identity ──
    d["name"]     = ov.get("Name", ov.get("Symbol", ""))
    d["sector"]   = ov.get("Sector", "N/A")
    d["industry"] = ov.get("Industry", "N/A")

    # ── Price ──
    # Chart mode: live price from GLOBAL_QUOTE
    # Lean/Full mode: 52-week average proxy from OVERVIEW (less precise but free)
    d["price"] = (
        safe_float(q.get("05. price"))                   # live tick (Chart mode)
        or safe_float(ov.get("AnalystTargetPrice"))       # NOT used — analyst target ≠ price
    )
    # Best free-tier price proxy without GLOBAL_QUOTE:
    # Use the 200-day MA from OVERVIEW as a rough current-price stand-in if needed.
    # In practice, the user sees "N/A" and understands they need Chart mode for price.
    if not d["price"]:
        # Try 200DayMovingAverage as a rough proxy (disclosed in OVERVIEW, no extra call)
        d["price"] = safe_float(ov.get("200DayMovingAverage"))
        d["price_is_approx"] = True
    else:
        d["price_is_approx"] = False

    d["shares"]     = safe_float(ov.get("SharesOutstanding"))
    d["market_cap"] = safe_float(ov.get("MarketCapitalization"))
    d["beta"]       = safe_float(ov.get("Beta"))

    # ── Income statement fields ──
    # INCOME_STATEMENT gives the most accurate annual figures.
    d["revenue"]       = safe_float(inc.get("totalRevenue"))      or safe_float(ov.get("RevenueTTM"))
    d["revenue_prior"] = safe_float(inc_prior.get("totalRevenue"))
    d["net_income"]    = safe_float(inc.get("netIncome"))         or safe_float(ov.get("NetIncomeTTM"))
    d["gross_profit"]  = safe_float(inc.get("grossProfit"))
    d["ebitda"]        = safe_float(ov.get("EBITDA"))             # only in OVERVIEW
    d["eps"]           = safe_float(ov.get("EPS"))

    # EBIT = operating income  (Ch.6: preferred starting point for EPV)
    d["ebit"] = (
        safe_float(inc.get("operatingIncome"))
        or safe_float(inc.get("ebit"))
    )
    if not d["ebit"] and d["revenue"] and ov.get("OperatingMarginTTM"):
        om = safe_float(ov.get("OperatingMarginTTM"))
        if om: d["ebit"] = d["revenue"] * om   # derived fallback

    # Effective tax rate
    income_tax = safe_float(inc.get("incomeTaxExpense"), 0) or 0
    pretax     = safe_float(inc.get("incomeBeforeTax"), 0) or 0
    if pretax and pretax > 0 and income_tax:
        d["tax_rate"] = min(max(income_tax / pretax, 0.10), 0.45)
    else:
        d["tax_rate"] = 0.21   # US statutory fallback

    # ── Balance sheet (Full/Chart mode — precise; Lean mode — OVERVIEW fallback) ──
    d["total_debt"] = (
        safe_float(bal.get("shortLongTermDebtTotal"))
        or (safe_float(bal.get("longTermDebt"), 0) or 0) + (safe_float(bal.get("shortTermDebt"), 0) or 0)
        # OVERVIEW does not publish total debt directly; we estimate from EV if needed
    )
    d["cash"] = (
        safe_float(bal.get("cashAndCashEquivalentsAtCarryingValue"))
        or safe_float(bal.get("cashAndShortTermInvestments"))
    )
    d["total_equity"] = safe_float(bal.get("totalShareholderEquity"))

    # Book value per share
    d["book_value"] = (
        safe_div(d["total_equity"], d["shares"]) if d["total_equity"] and d["shares"]
        else safe_float(ov.get("BookValue"))    # OVERVIEW fallback (always available)
    )

    # Reconstruct total equity from OVERVIEW BookValue if balance not fetched
    if not d["total_equity"] and d["book_value"] and d["shares"]:
        d["total_equity"] = d["book_value"] * d["shares"]

    # Lean-mode cash/debt estimate: use EV identity
    # EV = MktCap + Debt − Cash  →  Debt − Cash = EV − MktCap
    # OVERVIEW provides EVToEBITDA; we derive $ EV = EVToEBITDA × EBITDA
    if not d["cash"] and not d["total_debt"] and d["ebitda"] and ov.get("EVToEBITDA"):
        ev_ebitda = safe_float(ov.get("EVToEBITDA"))
        if ev_ebitda and d["ebitda"]:
            ev_dollars   = ev_ebitda * d["ebitda"]
            mktcap       = d["market_cap"] or 0
            net_debt_est = ev_dollars - mktcap   # Debt − Cash
            # Split roughly: if net_debt_est > 0 → net debt; < 0 → net cash
            if net_debt_est >= 0:
                d["total_debt"] = net_debt_est
                d["cash"]       = 0.0
            else:
                d["total_debt"] = 0.0
                d["cash"]       = abs(net_debt_est)
            d["debt_cash_estimated"] = True
    else:
        d["debt_cash_estimated"] = False

    # ── Cash flow (Full/Chart mode — precise; Lean mode — estimated) ──
    d["capex"]       = abs(safe_float(cf.get("capitalExpenditures"), 0) or 0)
    d["da"]          = safe_float(cf.get("depreciationDepletionAndAmortization"), 0) or 0
    d["operating_cf"]= safe_float(cf.get("operatingCashflow"), 0)
    d["dividends_paid"] = abs(safe_float(cf.get("dividendPayout"), 0) or 0)

    # Lean-mode D&A estimate: EBITDA − EBIT (by definition, EBITDA = EBIT + D&A)
    if not d["da"] and d["ebitda"] and d["ebit"]:
        d["da"] = max(d["ebitda"] - d["ebit"], 0)
        d["da_estimated"] = True
    else:
        d["da_estimated"] = False

    # Lean-mode capex estimate: 3% of revenue (rough industry average)
    if not d["capex"] and d["revenue"]:
        d["capex"] = d["revenue"] * 0.03
        d["capex_estimated"] = True
    else:
        d["capex_estimated"] = False

    # Dividend per share
    d["dividend_ttm"] = (
        safe_div(d["dividends_paid"], d["shares"]) if d["shares"] and d["dividends_paid"]
        else safe_float(ov.get("DividendPerShare"), 0)
    )

    # ── Multiples (OVERVIEW — always available) ──
    d["pe_ratio"]      = safe_float(ov.get("PERatio"))
    d["pb_ratio"]      = safe_float(ov.get("PriceToBookRatio"))
    d["ps_ratio"]      = safe_float(ov.get("PriceToSalesRatioTTM"))
    d["dividend_yield"]= safe_float(ov.get("DividendYield"), 0)

    # Enterprise value in dollars
    d["ev_dollars"] = (
        (d["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0)
        if d["market_cap"] else None
    )

    # ── Profitability (OVERVIEW — always available) ──
    d["roe"]              = safe_float(ov.get("ReturnOnEquityTTM"))
    d["roa"]              = safe_float(ov.get("ReturnOnAssetsTTM"))
    d["gross_margin"]     = safe_div(d["gross_profit"], d["revenue"])
    d["operating_margin"] = safe_float(ov.get("OperatingMarginTTM"))
    d["net_margin"]       = safe_float(ov.get("ProfitMargin"))

    # ── Growth ──
    if d["revenue"] and d["revenue_prior"] and d["revenue_prior"] != 0:
        d["revenue_growth"] = (d["revenue"] - d["revenue_prior"]) / abs(d["revenue_prior"])
    else:
        d["revenue_growth"] = safe_float(ov.get("QuarterlyRevenueGrowthYOY"))
    d["earnings_growth"] = safe_float(ov.get("QuarterlyEarningsGrowthYOY"))

    # ── ROIC — book-correct formula (Ch. 7) ──────────────────────────────────
    # NOPAT = EBIT × (1 − tax_rate)
    # Surplus Cash = Cash − 1% of Revenue  (Ch. 7: "1% of sales" standard)
    # Invested Capital = Equity + Debt − Surplus Cash
    # ROIC = NOPAT / Invested Capital
    try:
        op_cash_needed = (d["revenue"] or 0) * 0.01
        surplus_cash   = max((d["cash"] or 0) - op_cash_needed, 0)
        nopat          = (d["ebit"] or 0) * (1 - d["tax_rate"])
        inv_cap        = (d["total_equity"] or 0) + (d["total_debt"] or 0) - surplus_cash
        d["roic"]           = safe_div(nopat, inv_cap) if inv_cap and inv_cap > 0 else None
        d["surplus_cash"]   = surplus_cash
        d["nopat"]          = nopat
        d["invested_capital"] = inv_cap
    except:
        d["roic"] = None
        d["surplus_cash"] = d["nopat"] = d["invested_capital"] = 0

    # ── 1-year price history (Chart mode only) ──
    ts = raw["daily"].get("Time Series (Daily)", {})
    if ts:
        dates  = sorted(ts.keys())[-252:]
        prices = [safe_float(ts[dt]["4. close"]) for dt in dates]
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
    # Base = book equity; add LIFO reserve to convert inventory to replacement cost.
    book_total = (
        (d["book_value"] or 0) * (d["shares"] or 0)
        if d["book_value"] and d["shares"]
        else (d["total_equity"] or 0)
    )
    v["reproduction_cost"] = book_total + lifo_reserve

    # ── EPV (Ch. 3, 5, 6, 7) ─────────────────────────────────────────────────
    # Step 1: NOPAT = EBIT × (1 − tax_rate)
    # Step 2: Add 25% of D&A (conservative maintenance-capex buffer, Ch.7 Intel)
    # Step 3: Divide by R (zero-growth perpetuity)
    # Step 4: Subtract debt, add surplus cash → equity EPV
    if d["ebit"]:
        nopat_epv  = d["ebit"] * (1 - d["tax_rate"])
        da_addback = (d["da"] or 0) * 0.25
        adj_earn   = nopat_epv + da_addback
    else:
        adj_earn   = d["net_income"] or ((d["eps"] or 0) * (d["shares"] or 1))

    epv_enterprise      = safe_div(adj_earn, r)
    surplus             = d.get("surplus_cash") or 0
    v["epv_total"]      = (epv_enterprise - (d["total_debt"] or 0) + surplus) if epv_enterprise is not None else None
    v["epv_per_share"]  = safe_div(v["epv_total"], d["shares"])
    v["adj_earnings"]   = adj_earn

    # ── Margin of Safety ──────────────────────────────────────────────────────
    v["mos_per_share"] = (
        (v["epv_per_share"] or 0) - (d["price"] or 0)
        if v["epv_per_share"] and d["price"] else None
    )
    v["mos_total"]  = (v["epv_total"] or 0) - (v["market_cap"] or 0) if v["epv_total"] else None
    v["mos_pct"]    = safe_div(v["mos_per_share"], v["epv_per_share"]) \
                      if v["epv_per_share"] and v["epv_per_share"] != 0 else None

    # ── Franchise Value (Ch. 5) ───────────────────────────────────────────────
    v["franchise_value"]     = (v["epv_total"] or 0) - v["reproduction_cost"] if v["epv_total"] else None
    v["franchise_per_share"] = safe_div(v["franchise_value"], d["shares"])

    # ── PV with Growth = C × (ROC − G) / (R − G)  (Ch. 7 Appendix) ──────────
    roc = d["roic"] or d["roe"] or 0
    v["growth_factor_F"] = safe_div(roc - g, rg) if rg > 0 and roc else None

    if rg > 0 and roc and d.get("invested_capital") and d["invested_capital"] > 0:
        pv_ent           = d["invested_capital"] * safe_div(roc - g, rg)
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
        "Alpha Vantage API Key", type="password",
        help="Free key at alphavantage.co/support/#api-key"
    )
    if not api_key:
        st.info("Get a free key at [alphavantage.co](https://www.alphavantage.co)")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("#### 📡 Data Mode")
    mode = st.radio(
        "API calls per analysis:",
        options=["Lean (2 calls)", "Full (4 calls)", "Chart (6 calls)"],
        index=1,
        help=(
            "Lean: OVERVIEW + INCOME — covers ~90% of metrics, D&A and debt estimated.\n"
            "Full: + BALANCE_SHEET + CASH_FLOW — precise debt, cash, capex, D&A.\n"
            "Chart: + GLOBAL_QUOTE + DAILY — live price + 1-year chart."
        )
    )

    # ── API call budget meter ─────────────────────────────────────────────────
    # Tracks real network calls made THIS session only (resets on page reload).
    # Re-running the same ticker costs ZERO calls — served from 8-hour cache.
    calls_this     = {"Lean (2 calls)": 2, "Full (4 calls)": 4, "Chart (6 calls)": 6}[mode]
    calls_used     = st.session_state.get("calls_used_session", 0)
    cached_tickers = st.session_state.get("cached_tickers", set())
    is_cached      = _cache_key(ticker_input.upper().strip(), mode) in cached_tickers
    budget_left    = max(25 - calls_used, 0)
    analyses_left  = budget_left // calls_this if calls_this else 0
    cache_note     = "⚡ Cached — 0 calls" if is_cached else f"📡 Fresh — {calls_this} calls"

    st.markdown(
        f'<div class="call-meter">'
        f'🔋 Session calls used: <b>{calls_used}</b> (resets on page reload)<br>'
        f'📊 Next run: <b>{cache_note}</b><br>'
        f'🔄 ~{analyses_left} fresh analyses left this session<br>'
        f'<span style="color:#888;font-size:11px">Free tier: 25 calls/day total across all sessions</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    if budget_left <= 4 and not is_cached:
        st.warning("⚠️ Running low — switch to Lean mode or re-run cached tickers.")

    st.markdown("#### Assumptions")
    r = st.slider("Cost of Capital (R)", 0.04, 0.20, 0.09, 0.005, format="%.3f",
                  help="Ch. 7: 'We think 12% is a reasonable number — several percentages higher than the long-term S&P 500 return.'")
    g = st.slider("Perpetual Growth Rate (G)", 0.00, 0.08, 0.03, 0.005, format="%.3f",
                  help="Ch. 7: 'If G exceeds R, the value of the firm is infinite — so R is the realistic limit.'")
    lifo_reserve = st.number_input(
        "LIFO Reserve ($M)", min_value=0.0, value=0.0, step=10.0, format="%.1f",
        help="Ch. 4: Add the LIFO reserve from the 10-K footnotes to adjust inventory to replacement cost."
    ) * 1e6

    st.divider()
    run = st.button("🔍 Analyse", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 Value Investing Toolkit")
st.caption("Based on *Value Investing: From Graham to Buffett and Beyond* (Greenwald et al.)")

if g >= r:
    st.error("⚠️ G must be < R. (Ch. 7: if G ≥ R, value becomes infinite.)")
    st.stop()
if not api_key:
    st.warning("Enter your Alpha Vantage API key in the sidebar.")
    st.stop()
if not run and "last_ticker" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click **Analyse**.")
    st.stop()

ticker = ticker_input if run else st.session_state.get("last_ticker", ticker_input)
if run:
    st.session_state["last_ticker"] = ticker_input
    st.session_state["last_mode"]   = mode
    # Track calls only for genuinely new (ticker, mode) fetches.
    # Same ticker re-run = served from av_get() 8-hour cache = 0 real calls.
    # REMOVED: st.cache_data.clear() — this was burning the daily budget
    #   by forcing fresh API calls on every single Analyse click.
    ck = _cache_key(ticker_input.upper().strip(), mode)
    known = st.session_state.get("cached_tickers", set())
    if ck not in known:
        st.session_state["calls_used_session"] = (
            st.session_state.get("calls_used_session", 0) + calls_this
        )
        known.add(ck)
        st.session_state["cached_tickers"] = known

mode_used = mode if run else st.session_state.get("last_mode", mode)

# ── Fetch ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} ({mode_used}) …"):
    try:
        raw = fetch_data(ticker, api_key, mode_used)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Network error: {e}")
        st.stop()

if not raw["overview"] or not raw["overview"].get("Symbol"):
    st.error(f"No data for **{ticker}**. Check ticker and API key.")
    st.stop()

d = extract_financials(raw)
v = compute_valuations(d, r, g, lifo_reserve)

# ── Data quality banner ───────────────────────────────────────────────────────
lean = mode_used == "Lean (2 calls)"
warnings = []
if lean:
    warnings.append("**Lean mode:** balance sheet (debt/cash) and cash flow (D&A/capex) are estimated.")
if d.get("debt_cash_estimated"):
    warnings.append("Debt & cash estimated from EV/EBITDA ratio — upgrade to Full for precision.")
if d.get("da_estimated"):
    warnings.append("D&A estimated as EBITDA − EBIT — upgrade to Full for precision.")
if d.get("capex_estimated"):
    warnings.append("Capex estimated at 3% of revenue — upgrade to Full for precision.")
if d.get("price_is_approx") and d.get("price"):
    warnings.append("Price shown is 200-day moving average (proxy) — upgrade to Chart for live price.")
if not d.get("price"):
    warnings.append("Price unavailable in Lean/Full mode — upgrade to Chart mode for live price and MoS.")
if warnings:
    with st.expander(f"⚠️ {len(warnings)} data quality note(s) — click to expand", expanded=lean):
        for w in warnings:
            st.caption(f"• {w}")

# ── Company header ─────────────────────────────────────────────────────────────
col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
with col_a:
    st.subheader(f"{d['name']}  [{ticker}]")
    st.caption(f"{d['sector']}  ·  {d['industry']}  ·  Mode: **{mode_used}** ({calls_this} calls used)")
with col_b:
    price_label = f"${d['price']:,.2f}" if d["price"] else "N/A"
    if d.get("price_is_approx") and d.get("price"):
        price_label += " ~"
    st.metric("Price", price_label,
              help="~ = 200-day MA proxy. Use Chart mode for live price.")
with col_c:
    st.metric("Market Cap", fmt_currency(v["market_cap"]))
with col_d:
    st.metric("1Y Return", fmt_pct(v["total_return"]) if v["total_return"] else "N/A (Chart mode)")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Overview", "💰 Valuation (EPV)", "🚀 Growth Analysis", "🏭 Asset & Franchise", "🎯 Summary"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(book_note(
        "Ch. 2: The market overview provides context, not valuation. Market price may be wrong. "
        "The value investor's job is to determine intrinsic value independently, then compare."
    ), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enterprise Value",  fmt_currency(v["ev"]),
              help="EV = Market Cap + Debt − Cash (Ch. 4)")
    c2.metric("P/E Ratio",         fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A")
    c3.metric("P/B Ratio",         fmt_x(d["pb_ratio"]) if d["pb_ratio"] else "N/A",
              help="If P/B > 1, the market values the franchise above asset reproduction cost.")
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
        st.info("📡 Price chart requires **Chart (6 calls)** mode. Switch in the sidebar.")

    st.markdown('<div class="section-header">Profitability</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 6: Stable, high operating margins over multiple years signal a franchise. "
        "WD-40's consistent 24–27% EBIT margin with no R&D moat was the key tell. "
        "Look for margin stability, not just level."
    ), unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Gross Margin",      fmt_pct(d["gross_margin"]))
    pc2.metric("Operating Margin",  fmt_pct(d["operating_margin"]),
               help="EBIT ÷ Revenue — the book's preferred profitability measure")
    pc3.metric("Net Margin",        fmt_pct(d["net_margin"]))
    pc4.metric("ROE",               fmt_pct(d["roe"]))

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Earnings Power Value (EPV) — Greenwald</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 3: EPV = Adjusted Earnings ÷ R. Zero-growth, most conservative earnings estimate. "
        "Ch. 6/7: Start with EBIT (not net income), apply tax, add back 25% of D&A as a "
        "conservative maintenance-capex buffer, then subtract debt and add surplus cash."
    ), unsafe_allow_html=True)

    with st.expander("📐 EPV Calculation Breakdown", expanded=True):
        eb1, eb2, eb3 = st.columns(3)
        eb1.metric("EBIT (Operating Income)", fmt_currency(d["ebit"]),
                   help="Ch. 6: Start here, not at net income. Excludes interest (capital-structure choice).")
        eb2.metric(f"Tax Rate", fmt_pct(d["tax_rate"]))
        eb3.metric("NOPAT = EBIT × (1−tax)", fmt_currency(d.get("nopat")))

        eb4, eb5, eb6 = st.columns(3)
        da_add = (d["da"] or 0) * 0.25
        da_lbl = "D&A Addback 25%" + (" (est.)" if d.get("da_estimated") else "")
        eb4.metric(da_lbl, fmt_currency(da_add),
                   help="Ch. 7 Intel: 'add back 25% of D&A — the other 75% covers maintenance capex'")
        eb5.metric("Surplus Cash Added", fmt_currency(d.get("surplus_cash")),
                   help="Ch. 7: Cash > 1% of sales is surplus. Added to equity EPV.")
        eb6.metric("Debt Subtracted", fmt_currency(d.get("total_debt")),
                   help="Ch. 7: Subtract interest-bearing debt to convert enterprise EPV to equity EPV.")

        st.caption(
            f"EPV (equity) = (NOPAT + 25% D&A) ÷ R − Debt + Surplus Cash"
        )

    e1, e2, e3 = st.columns(3)
    e1.metric("EPV (Total Equity)",  fmt_currency(v["epv_total"]))
    e2.metric("EPV per Share",       f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    e3.metric("Current Price",
              (f"${d['price']:,.2f}" + (" ~" if d.get("price_is_approx") else ""))
              if d["price"] else "N/A (use Chart mode)")

    st.markdown('<div class="section-header">Margin of Safety</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Graham / Ch. 3: MoS = (EPV − Price) / EPV. "
        "≥33% = adequate protection (BUY). ≥50% = strong buy. "
        "Negative MoS = market is pricing in growth. Only justified with a confirmed franchise."
    ), unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("MoS per Share",  f"${v['mos_per_share']:,.2f}" if v["mos_per_share"] else "N/A",
              delta=fmt_pct(v["mos_pct"]) if v["mos_pct"] else None)
    m2.metric("MoS Total",      fmt_currency(v["mos_total"]))
    m3.metric("MoS %",          fmt_pct(v["mos_pct"]))
    if not d.get("price"):
        st.info("📡 Margin of Safety requires a price. Switch to **Chart (6 calls)** mode.")
    else:
        st.markdown("**Signal:** " + mos_signal(v["mos_pct"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dividend Discount Model (DDM) — Ch. 6</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "V = Dividend ÷ (R − G). Applied in Ch. 6 (WD-40) as a cross-check on EPV "
        "for dividend-paying franchises. Only reliable when dividends represent true "
        "distributable cash flow and are paid consistently."
    ), unsafe_allow_html=True)
    if d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        d1, d2, d3 = st.columns(3)
        d1.metric("Dividend / Share (TTM)", f"${d['dividend_ttm']:,.4f}")
        d2.metric("DDM Value / Share",      f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A")
        d3.metric("DDM vs Price",
                  f"${(v['ddm_per_share'] or 0) - (d['price'] or 0):+,.2f}"
                  if v["ddm_per_share"] and d["price"] else "N/A")
    else:
        st.warning("No dividends paid — DDM not applicable.")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Growth Factor F & PV with Growth — Ch. 7</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7 Appendix: PV = C × (ROC − G) / (R − G)   where C = Invested Capital. "
        "F = (ROC − G) / (R − G) is the Growth Factor. "
        "F < 1 = growth destroys value. F = 1 = neutral. F > 1 = growth creates value. "
        "'Growth only adds value when ROC > R.' — Ch. 7"
    ), unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROIC",              fmt_pct(d["roic"]),
              help="NOPAT ÷ (Equity + Debt − Surplus Cash). Ch. 7 formula.")
    g2.metric("ROE",               fmt_pct(d["roe"]),
              help="Net Income ÷ Book Equity. Fallback when ROIC unavailable.")
    g3.metric("Growth Factor F",   fmt_x(v["growth_factor_F"]),
              help="F = (ROC − G) / (R − G). F > 1 = growth creates value.")
    g4.metric("Growth Mult. M",    fmt_x(v["growth_mult_M"]),
              help="M = PV / EPV. How much growth adds above EPV.")

    roc_used = d["roic"] or d["roe"]
    if roc_used and v["growth_factor_F"]:
        if roc_used > r:
            st.success(f"✅ ROC ({fmt_pct(roc_used)}) > R ({fmt_pct(r)}) — growth creates value (F = {fmt_x(v['growth_factor_F'])}).")
        elif abs(roc_used - r) < 0.01:
            st.warning("⚠️ ROC ≈ R — growth is value-neutral.")
        else:
            st.error(f"🔴 ROC ({fmt_pct(roc_used)}) < R ({fmt_pct(r)}) — growth destroys value (F = {fmt_x(v['growth_factor_F'])}).")

    st.markdown('<div class="section-header">PV with Growth vs EPV</div>', unsafe_allow_html=True)
    pv1, pv2, pv3 = st.columns(3)
    pv1.metric("EPV / Share (zero growth)",  f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    pv2.metric("PV / Share (with growth)",   f"${v['pv_per_share']:,.2f}"  if v["pv_per_share"]  else "N/A",
               help="C × (ROC−G)/(R−G), equity-adjusted. Ch. 7 exact formula.")
    pv3.metric("Growth Premium",
               fmt_currency((v["pv_with_growth"] or 0) - (v["epv_total"] or 0))
               if v["pv_with_growth"] and v["epv_total"] else "N/A")

    st.markdown('<div class="section-header">Growth Value Matrix (Book Table 7.11)</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7: PV/EPV ratios for combinations of ROC/R and G/R. "
        "It takes BOTH high ROC relative to R AND high G relative to R to generate meaningful "
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
    tr3.metric("Revenue (TTM)",         fmt_currency(d["revenue"]))

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Three-Slice Valuation Summary (Ch. 3, Fig. 3.1)</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Safety hierarchy: Asset Value (most reliable) → EPV → PV with Growth (least reliable). "
        "A safe buy: Price < EPV. If EPV > Asset Value → franchise evidence. "
        "If Price > EPV → market pricing in growth (only justified with confirmed moat)."
    ), unsafe_allow_html=True)
    sv1, sv2, sv3, sv4 = st.columns(4)
    sv1.metric("① Asset Value",        fmt_currency(v["reproduction_cost"]),
               help="Book equity + LIFO reserve. Floor valuation. Ch. 4.")
    sv2.metric("② EPV (Equity)",        fmt_currency(v["epv_total"]),
               help="NOPAT/R, debt/cash adjusted. Zero growth. Ch. 3, 5.")
    sv3.metric("③ PV with Growth",      fmt_currency(v["pv_with_growth"]) if v["pv_with_growth"] else "N/A",
               help="C × (ROC−G)/(R−G). Only valid if franchise confirmed. Ch. 7.")
    sv4.metric("Market Cap",            fmt_currency(v["market_cap"]))

    if v["epv_total"] and v["reproduction_cost"]:
        ratio = safe_div(v["epv_total"], v["reproduction_cost"])
        if ratio and ratio > 1:
            st.success(f"✅ EPV ({fmt_currency(v['epv_total'])}) > Reproduction Cost ({fmt_currency(v['reproduction_cost'])}) by {fmt_x(ratio)} — franchise value confirmed.")
        elif ratio and ratio < 0.8:
            st.error("🔴 EPV < Reproduction Cost — poor management or declining industry (Ch. 3).")
        else:
            st.warning("⚠️ EPV ≈ Reproduction Cost — no clear franchise.")

    st.markdown('<div class="section-header">Franchise Value — Ch. 5</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 5: Franchise Value = EPV − Reproduction Cost. "
        "Positive → the company earns above a competitive return on its assets → moat. "
        "'Without barriers to entry, competition forces returns down to cost of capital.' — Ch. 5"
    ), unsafe_allow_html=True)
    fv1, fv2, fv3 = st.columns(3)
    fv1.metric("EPV (Equity)",        fmt_currency(v["epv_total"]))
    fv2.metric("Reproduction Cost",   fmt_currency(v["reproduction_cost"]),
               help="Book equity + LIFO reserve. Ch. 4.")
    fv3.metric("Franchise Value",     fmt_currency(v["franchise_value"]),
               help="EPV − Reproduction Cost. Positive = moat.")
    if v["franchise_value"] and v["epv_total"]:
        fv_pct = safe_div(v["franchise_value"], v["epv_total"])
        if (v["franchise_value"] or 0) > 0:
            st.success(f"✅ Franchise value = {fmt_currency(v['franchise_value'])} ({fmt_pct(fv_pct)} of EPV).")
        else:
            st.warning("⚠️ No franchise value — EPV ≤ Reproduction Cost.")

    st.markdown('<div class="section-header">ROIC Components — Ch. 7</div>', unsafe_allow_html=True)
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("NOPAT",             fmt_currency(d.get("nopat")))
    rc2.metric("Invested Capital",  fmt_currency(d.get("invested_capital")))
    rc3.metric("Surplus Cash",      fmt_currency(d.get("surplus_cash")))
    rc4.metric("ROIC",              fmt_pct(d["roic"]))
    if lean:
        st.caption("⚠️ Invested capital uses estimated debt/cash in Lean mode. Use Full mode for precision.")

    st.markdown('<div class="section-header">Cap Rate — Ch. 10 (Gabelli)</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 10: Cap Rate = (EBITDA − Capex) ÷ Enterprise Value. "
        "Gabelli's Private Market Value approach. A cap rate > 8% = attractively priced."
        + (" Capex is estimated at 3% of revenue in Lean mode." if d.get("capex_estimated") else "")
    ), unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("EBITDA − Capex",   fmt_currency(v["operating_cf_for_cap"]))
    ca2.metric("Enterprise Value", fmt_currency(v["ev"]))
    ca3.metric("Cap Rate",         fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A")
    st.markdown("**Signal:** " + cap_signal(v["cap_rate"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sonkin Adjusted P/E — Ch. 16</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 16: Strip net cash from both market cap and earnings. "
        "Adjusted P/E = (Market Cap − Net Cash) ÷ (Net Income − Interest on Surplus Cash). "
        "Prevents value traps where cash inflates apparent cheapness."
    ), unsafe_allow_html=True)
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Net Cash",         fmt_currency(v["net_cash"]))
    sp2.metric("Op. Market Cap",   fmt_currency((v["market_cap"] or 0) - v["net_cash"]))
    sp3.metric("Sonkin Adj. P/E",  fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 🎯 Summary: {d['name']} [{ticker}]")
    st.caption(f"R = {fmt_pct(r)}  |  G = {fmt_pct(g)}  |  Mode: {mode_used}  |  {pd.Timestamp.now().strftime('%d %b %Y')}")

    summary_rows = [
        ("Market",    "Price",                  (f"${d['price']:,.2f}" + (" ~" if d.get("price_is_approx") else "")) if d["price"] else "N/A", "—"),
        ("Market",    "Market Cap",              fmt_currency(v["market_cap"]),    "—"),
        ("Market",    "Enterprise Value",        fmt_currency(v["ev"]),            "—"),
        ("Market",    "1Y Total Return",         fmt_pct(v["total_return"]) if v["total_return"] else "N/A (Chart mode)", "—"),
        ("Valuation", "① Reproduction Cost",    fmt_currency(v["reproduction_cost"]), "—"),
        ("Valuation", "② EPV / Share",           f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A", "—"),
        ("Valuation", "③ PV w/ Growth / Share",  f"${v['pv_per_share']:,.2f}"  if v["pv_per_share"]  else "N/A", "—"),
        ("Valuation", "Margin of Safety",        fmt_pct(v["mos_pct"]) if v["mos_pct"] else "N/A",
                                                 "✅ BUY" if (v["mos_pct"] or 0) >= 0.33
                                                 else ("⚠️ HOLD" if (v["mos_pct"] or 0) >= 0.10 else "🔴 SELL")),
        ("Valuation", "DDM / Share",             f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A", "—"),
        ("Valuation", "P/E Ratio",               fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A", "—"),
        ("Growth",    "ROIC",                    fmt_pct(d["roic"]) if d["roic"] else "N/A",
                                                 "✅ ROC>R" if (d["roic"] or 0) > r else ("⚠️ ROC≈R" if abs((d["roic"] or 0) - r) < 0.02 else "🔴 ROC<R")),
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
        "MoS is your protection against estimation errors. "
        "'The greater the margin of safety, the lower the risk.' — Ch. 3"
    ), unsafe_allow_html=True)
    st.caption("⚠️ Research only — not financial advice. Verify all data independently.")
  
