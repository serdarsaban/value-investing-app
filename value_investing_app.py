"""
Value Investing Toolkit — Streamlit App
Based on: Value Investing: From Graham to Buffett and Beyond (Greenwald, Kahn, Sonkin, van Biema)
Data: Alpha Vantage API (free tier — OVERVIEW + INCOME_STATEMENT + BALANCE_SHEET + CASH_FLOW)

═══════════════════════════════════════════════════════════════════════════════
BOOK OVERVIEW — THREE SLICES OF VALUE (Ch. 3, Figure 3.1)
═══════════════════════════════════════════════════════════════════════════════
The Greenwald framework always starts with the most reliable estimate and
only moves to less certain ones when the data supports it:

  Slice 1 — ASSET VALUE (Reproduction Cost of Assets)
    "The first step in a Graham and Dodd valuation is to calculate the asset
     value of the company." — Ch. 4
    What would it cost a new competitor to replicate this business from scratch?
    Uses book value as a starting point, adjusted for LIFO reserves and other
    accounting distortions.

  Slice 2 — EARNINGS POWER VALUE (EPV)
    "EPV = Adjusted Earnings × 1/R" — Ch. 3
    What is the company worth if its current earnings continue forever at this
    level, with zero growth? Uses EBIT-based adjusted earnings, not net income,
    to remove capital-structure distortions (interest, excess cash).

  Slice 3 — VALUE WITH GROWTH (PV)
    "PV = C × (ROC − G) / (R − G)" — Ch. 7 Appendix
    Growth only creates value when ROC > R. This is the riskiest estimate and
    should only be used when a durable franchise/moat is confirmed.

Safety hierarchy: Asset Value < EPV < PV with Growth
The larger EPV is vs. Asset Value, the stronger the franchise evidence.
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
  .book-note {
    background: #1a1a2e; border-left: 3px solid #ffd740;
    padding: 10px 14px; border-radius: 4px;
    font-size: 13px; color: #ccc; margin: 8px 0 14px 0;
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

def book_note(text):
    """Render a yellow-left-border book quote box."""
    return f'<div class="book-note">📖 {text}</div>'


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
    """
    Fetch all required endpoints. We use 6 calls per ticker:
      OVERVIEW, INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW,
      GLOBAL_QUOTE, TIME_SERIES_DAILY
    Free tier allows 25 calls/day and 5/min.
    """
    overview  = av_get("OVERVIEW",          ticker, api_key)
    income    = av_get("INCOME_STATEMENT",  ticker, api_key)
    balance   = av_get("BALANCE_SHEET",     ticker, api_key)
    cashflow  = av_get("CASH_FLOW",         ticker, api_key)
    quote     = av_get("GLOBAL_QUOTE",      ticker, api_key)
    daily     = av_get("TIME_SERIES_DAILY", ticker, api_key, outputsize="compact")
    return overview, income, balance, cashflow, quote, daily


# ── Extract financials from AV responses ──────────────────────────────────────

def latest(report_list):
    """Return the most recent annual report dict from AV annualReports list."""
    try:
        return report_list[0] if report_list else {}
    except Exception:
        return {}

def prior(report_list):
    """Return the second-most-recent annual report (for YOY growth)."""
    try:
        return report_list[1] if len(report_list) > 1 else {}
    except Exception:
        return {}

def extract_financials(overview, income, balance, cashflow, quote, daily):
    """
    Pull the raw financial data needed for all Greenwald valuations.

    KEY PRINCIPLE (Ch. 6, 7): The book prefers working from EBIT (operating
    income) downward rather than from net income upward, because net income
    includes interest income/expense and other capital-structure items that
    distort the true operating earnings power of the business.
    """
    d = {}

    q   = quote.get("Global Quote", {})
    inc = latest(income.get("annualReports", []))
    inc_prior = prior(income.get("annualReports", []))
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
    d["revenue"]      = safe_float(inc.get("totalRevenue"))
    d["revenue_prior"]= safe_float(inc_prior.get("totalRevenue"))
    d["net_income"]   = safe_float(inc.get("netIncome"))
    d["ebitda"]       = safe_float(overview.get("EBITDA"))
    d["eps"]          = safe_float(overview.get("EPS"))

    # ── EBIT (Operating Income) — the book's preferred earnings base ──
    # Ch. 6: "We prefer the second approach, which is to start with operating
    # income, or earnings before interest and taxes, and work down, calculating
    # the taxes that would be paid on operating income."
    # AV field: "operatingIncome" from annual income statement
    d["ebit"]         = safe_float(inc.get("operatingIncome")) or \
                        safe_float(inc.get("ebit"))
    # Fallback: derive from operating margin × revenue
    if not d["ebit"] and d["revenue"] and overview.get("OperatingMarginTTM"):
        om = safe_float(overview.get("OperatingMarginTTM"))
        if om:
            d["ebit"] = d["revenue"] * om

    # Effective tax rate — from income statement if possible
    # Ch. 7 Intel: "we will assume a tax rate of 38 percent"
    # We compute it from reported figures; fall back to 21% (US corp. rate)
    income_tax = safe_float(inc.get("incomeTaxExpense"), 0) or 0
    pretax     = safe_float(inc.get("incomeBeforeTax"), 0) or 0
    if pretax and pretax > 0 and income_tax:
        d["tax_rate"] = min(max(income_tax / pretax, 0.10), 0.45)
    else:
        d["tax_rate"] = 0.21

    # D&A — needed for EPV adjustment
    # Ch. 7: "adjusting for depreciation, amortization, and capital expenditures"
    d["da"] = safe_float(cf.get("depreciationDepletionAndAmortization"), 0) or 0

    # ── Balance sheet ──
    d["total_debt"]   = safe_float(bal.get("shortLongTermDebtTotal")) or \
                        (safe_float(bal.get("longTermDebt"), 0) or 0) + \
                        (safe_float(bal.get("shortTermDebt"), 0) or 0)
    d["cash"]         = safe_float(bal.get("cashAndCashEquivalentsAtCarryingValue")) or \
                        safe_float(bal.get("cashAndShortTermInvestments"), 0)
    d["total_equity"] = safe_float(bal.get("totalShareholderEquity"))
    d["book_value"]   = safe_div(d["total_equity"], d["shares"]) \
                        if d["total_equity"] and d["shares"] else \
                        safe_float(overview.get("BookValue"))

    # ── Cash flow ──
    d["capex"]        = abs(safe_float(cf.get("capitalExpenditures"), 0) or 0)
    d["operating_cf"] = safe_float(cf.get("operatingCashflow"), 0)
    d["dividends_paid"]= abs(safe_float(cf.get("dividendPayout"), 0) or 0)

    # Dividend per share approximation
    d["dividend_ttm"] = safe_div(d["dividends_paid"], d["shares"]) if d["shares"] else \
                        safe_float(overview.get("DividendPerShare"), 0)

    # ── Multiples from overview ──
    d["pe_ratio"]     = safe_float(overview.get("PERatio"))
    d["pb_ratio"]     = safe_float(overview.get("PriceToBookRatio"))
    d["ps_ratio"]     = safe_float(overview.get("PriceToSalesRatioTTM"))
    d["ev_dollars"]   = (d["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0) \
                        if d["market_cap"] else None
    d["dividend_yield"]= safe_float(overview.get("DividendYield"), 0)

    # ── Growth (YOY from annual reports — more reliable than quarterly) ──
    if d["revenue"] and d["revenue_prior"] and d["revenue_prior"] != 0:
        d["revenue_growth"] = (d["revenue"] - d["revenue_prior"]) / abs(d["revenue_prior"])
    else:
        d["revenue_growth"] = safe_float(overview.get("QuarterlyRevenueGrowthYOY"))

    d["earnings_growth"] = safe_float(overview.get("QuarterlyEarningsGrowthYOY"))

    # ── Profitability ──
    d["roe"]              = safe_float(overview.get("ReturnOnEquityTTM"))
    d["roa"]              = safe_float(overview.get("ReturnOnAssetsTTM"))
    d["gross_margin"]     = safe_div(safe_float(inc.get("grossProfit")), d["revenue"])
    d["operating_margin"] = safe_float(overview.get("OperatingMarginTTM"))
    d["net_margin"]       = safe_float(overview.get("ProfitMargin"))

    # ── ROIC — Book-correct formula (Ch. 7) ──────────────────────────────────
    # "A measure favored by professionals is return on invested capital (ROIC),
    #  which uses operating earnings as the numerator and operating assets as
    #  the denominator." — Ch. 7
    #
    # Formula:
    #   NOPAT    = EBIT × (1 − tax_rate)
    #   Op.Assets = Total Equity + Total Debt − Surplus Cash
    #   Surplus Cash = Cash − 1% of Revenue  (book standard: Ch. 7)
    #   ROIC     = NOPAT / Op.Assets
    #
    # Why subtract surplus cash? The book says cash in excess of ~1% of sales
    # earns only interest, not operating returns. Including it would understate
    # ROIC. We strip it out to get the true return on capital actually deployed
    # in the business.
    try:
        operating_cash_needed = (d["revenue"] or 0) * 0.01  # Ch. 7: ~1% of sales
        surplus_cash = max((d["cash"] or 0) - operating_cash_needed, 0)
        nopat = (d["ebit"] or 0) * (1 - d["tax_rate"])
        invested_capital = (d["total_equity"] or 0) + (d["total_debt"] or 0) - surplus_cash
        d["roic"] = safe_div(nopat, invested_capital) if invested_capital > 0 else None
        d["surplus_cash"] = surplus_cash
        d["nopat"] = nopat
        d["invested_capital"] = invested_capital
    except Exception:
        d["roic"] = None
        d["surplus_cash"] = 0
        d["nopat"] = None
        d["invested_capital"] = None

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
    """
    All three Greenwald valuations plus supporting metrics.

    Parameters
    ----------
    d : dict of extracted financials
    r : float  Cost of capital (WACC) set by user
    g : float  Perpetual growth rate set by user
    lifo_reserve : float  LIFO inventory reserve in dollars (user input, Ch. 4)
    """
    v = {}
    rg = r - g  # The denominator used in all Gordon-Growth formulas

    # ─── Market metrics ───────────────────────────────────────────────────────
    v["market_cap"] = d["market_cap"] or \
                      ((d["price"] or 0) * (d["shares"] or 0) if d["price"] and d["shares"] else None)

    # Enterprise Value = Market Cap + Debt − Cash
    # Ch. 4: "Using what is called the enterprise value approach, we add the
    # market value of the debt to the market value of the equity and then
    # subtract cash."
    v["ev"] = d["ev_dollars"] or \
              ((v["market_cap"] or 0) + (d["total_debt"] or 0) - (d["cash"] or 0))

    # ─── SLICE 1: Asset Value / Reproduction Cost ─────────────────────────────
    # Ch. 4: "The first step in a Graham and Dodd valuation is to calculate
    # the asset value of the company… the economic value of the assets is their
    # reproduction costs—that is, what it would cost a would-be competitor to
    # get into this business."
    #
    # Full reproduction cost requires line-by-line balance sheet analysis.
    # Without that detail, we use Total Equity (book value) as the base and
    # add the LIFO reserve, which converts LIFO inventory to FIFO/replacement
    # cost — the single most common and material adjustment available from
    # public filings.
    #
    # Ch. 4: "Inventories: Add LIFO reserve, if any; adjust for turnover"
    # (Table 4.3 shows LIFO reserve as a standard line item in the asset table)
    #
    # NOTE: Book value already nets out liabilities, giving us the equity
    # reproduction cost directly. Debt holders have their own claim; we are
    # computing the equity slice.
    book_total = (d["book_value"] or 0) * (d["shares"] or 0) \
                 if d["book_value"] and d["shares"] else (d["total_equity"] or 0)
    v["reproduction_cost"] = book_total + lifo_reserve

    # ─── SLICE 2: Earnings Power Value (EPV) — book's central calculation ────
    # Ch. 3: "EPV = Adjusted Earnings × 1/R"
    # Ch. 6: "we prefer the second approach, which is to start with operating
    # income [EBIT], and work down, calculating the taxes that would be paid
    # on operating income and adjusting for depreciation and capex."
    #
    # Correct Adjusted Earnings formula:
    #   Step 1: EBIT × (1 − tax_rate) = NOPAT (after-tax operating earnings)
    #   Step 2: Add back 25% of D&A  [Ch. 7: "add back 25% of D&A, assuming
    #           the other 75% will more than cover maintenance capex"]
    #           This 25% adjustment handles the depreciation-vs-maintenance-
    #           capex gap conservatively.
    #   Step 3: Adjust for surplus cash/debt to get to equity EPV
    #           Ch. 7: "subtract the book value of interest-bearing debt and
    #           add back all cash in excess of 1% of sales"
    #
    # WHY NOT NET INCOME?
    # Net income includes interest expense (a capital structure choice, not an
    # operating fact) and interest income from cash (which we handle separately).
    # Using EBIT makes the EPV comparable across companies with different
    # capital structures and allows a clean debt/cash adjustment at the end.

    if d["ebit"]:
        nopat_for_epv = d["ebit"] * (1 - d["tax_rate"])
        # Add back 25% of D&A as conservative maintenance capex adjustment (Ch. 7)
        da_addback = (d["da"] or 0) * 0.25
        adj_earnings = nopat_for_epv + da_addback
    else:
        # Fallback to net income if EBIT not available (less accurate per book)
        adj_earnings = d["net_income"] or ((d["eps"] or 0) * (d["shares"] or 1))

    # EPV of the whole enterprise (debt + equity)
    epv_enterprise = safe_div(adj_earnings, r)

    # Convert to equity EPV: subtract debt, add surplus cash
    # Ch. 7: "reduce EPV by the amount of the debt outstanding… add back all
    # cash in excess of 1% of sales"
    if epv_enterprise is not None:
        surplus = d.get("surplus_cash", 0) or 0
        v["epv_total"] = epv_enterprise - (d["total_debt"] or 0) + surplus
    else:
        v["epv_total"] = None

    v["epv_per_share"]  = safe_div(v["epv_total"], d["shares"])
    v["adj_earnings"]   = adj_earnings  # expose for display

    # ─── Margin of Safety (vs EPV) ────────────────────────────────────────────
    # Ch. 3 / Graham: Buy when price is meaningfully below intrinsic value.
    # The book uses EPV as the conservative intrinsic value (no growth assumed).
    # Graham's guideline: ≥33% MoS for adequate protection; ≥50% for strong buy.
    #
    # MoS = (EPV − Price) / EPV
    # Positive MoS → stock trading below EPV → potential buy
    # Negative MoS → stock trading above EPV → growth premium priced in
    v["mos_per_share"] = ((v["epv_per_share"] or 0) - (d["price"] or 0)) \
                         if v["epv_per_share"] and d["price"] else None
    v["mos_total"]     = (v["epv_total"] or 0) - (v["market_cap"] or 0) \
                         if v["epv_total"] else None
    v["mos_pct"]       = safe_div(v["mos_per_share"], v["epv_per_share"]) \
                         if v["epv_per_share"] and v["epv_per_share"] != 0 else None

    # ─── Franchise Value = EPV − Reproduction Cost ───────────────────────────
    # Ch. 5: "The $60 million discrepancy here between the asset value and the
    # EPV… explains [the franchise]. If EPV > Reproduction Cost, the company
    # is earning more than a competitive return on its assets — evidence of a
    # durable competitive advantage (moat)."
    # Ch. 3: "There are only two conditions in which EPV < Asset Value: either
    # management is doing a poor job, or the industry is in decline."
    v["franchise_value"]     = (v["epv_total"] or 0) - v["reproduction_cost"] \
                               if v["epv_total"] else None
    v["franchise_per_share"] = safe_div(v["franchise_value"], d["shares"])

    # ─── SLICE 3: PV with Growth = C × (ROC − G) / (R − G) ──────────────────
    # Ch. 7 Appendix (the definitive formula):
    #   PV = C × (ROC − G) / (R − G)
    # where:
    #   C   = Invested Capital (equity + debt − surplus cash)
    #   ROC = Return on Invested Capital
    #   G   = perpetual growth rate (user input, must be < R)
    #   R   = cost of capital (user input)
    #
    # This simplifies to:  PV = EPV × F  where  F = (ROC−G)/(R−G)
    #   F = Growth Factor (F < 1 → growth destroys value, F = 1 → neutral,
    #                      F > 1 → growth creates value)
    #
    # IMPORTANT: PV here is an ENTERPRISE value, just like EPV before the
    # debt/cash adjustment. We apply the same debt/cash netting.
    #
    # Growth Multiplier M = PV / EPV = 1 − (G/R)(R/ROC) / 1 − (G/R)
    # From Ch. 7: "Substitution from these two expressions into the definition
    # of M and a little algebraic manipulation yield the equation
    # M = 1 − (G/R)(R/ROC) / 1 − (G/R)"

    roc = d["roic"] or d["roe"] or 0

    # Growth Factor F
    if rg > 0 and roc:
        v["growth_factor_F"] = safe_div(roc - g, rg)
    else:
        v["growth_factor_F"] = None

    # Full PV with Growth (enterprise, then equity-adjusted)
    if rg > 0 and roc and d.get("invested_capital") and d["invested_capital"] > 0:
        pv_enterprise = d["invested_capital"] * safe_div(roc - g, rg)
        surplus = d.get("surplus_cash", 0) or 0
        v["pv_with_growth"] = pv_enterprise - (d["total_debt"] or 0) + surplus \
                              if pv_enterprise else None
        v["pv_per_share"]   = safe_div(v["pv_with_growth"], d["shares"])
    else:
        v["pv_with_growth"] = None
        v["pv_per_share"]   = None

    # Growth Multiplier M = PV / EPV
    # Ch. 7: M = [1 − (G/R)(R/ROC)] / [1 − (G/R)]
    if r > 0 and roc and roc > 0:
        gr = g / r
        denom = 1 - gr * (r / roc)
        v["growth_mult_M"] = safe_div(1 - gr, denom) if denom != 0 else None
    else:
        v["growth_mult_M"] = None

    # ─── DDM — Dividend Discount Model ───────────────────────────────────────
    # Ch. 6 (WD-40 case): V = Dividend / (R − G)
    # This is a Gordon Growth Model. The book applies it to dividend-paying
    # franchises as a cross-check on EPV. It is only meaningful when:
    #   (a) the company pays a consistent dividend
    #   (b) dividends are a reliable proxy for distributable cash flow
    #   (c) G < R (already enforced in sidebar)
    if rg > 0 and d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        v["ddm_per_share"] = d["dividend_ttm"] / rg
        v["ddm_total"]     = v["ddm_per_share"] * (d["shares"] or 0)
    else:
        v["ddm_per_share"] = None
        v["ddm_total"]     = None

    # ─── Cap Rate — Gabelli / Ch. 10 ─────────────────────────────────────────
    # Mario Gabelli's "Private Market Value" approach:
    #   Cap Rate = (EBITDA − Capex) / Enterprise Value
    #
    # This is essentially a real-estate-style yield on the operating asset.
    # Gabelli uses it to ask: "What would a private acquirer pay for this
    # business, and what yield would they receive?"
    # A cap rate > 8% generally signals an attractively priced business.
    # Note: EBITDA − Capex ≈ unlevered free cash flow (a rough proxy).
    op_cf = (d["ebitda"] or 0) - (d["capex"] or 0)
    v["cap_rate"]             = safe_div(op_cf, v["ev"]) if v["ev"] else None
    v["operating_cf_for_cap"] = op_cf

    # ─── PEG Ratio — Ch. 11 (Glenn Greenberg) ────────────────────────────────
    # PEG = P/E ÷ Earnings Growth Rate (expressed as a percentage, not decimal)
    # Greenberg's rule of thumb: PEG < 1.0 signals undervaluation relative to
    # growth; > 3.0 signals the market is pricing in too much optimism.
    # The book emphasises: only use when the franchise is clear and growth is
    # within it — otherwise the P/E premium is unjustified.
    g_pct = (d["earnings_growth"] or d["revenue_growth"] or g) * 100
    v["peg"] = safe_div(d["pe_ratio"], g_pct) if d["pe_ratio"] and g_pct else None

    # ─── Sonkin Adjusted P/E — Ch. 16 ────────────────────────────────────────
    # Paul Sonkin's technique: strip out net cash/debt so you pay only for the
    # operating business. This prevents "value traps" where a low headline P/E
    # is inflated by cash that belongs to shareholders anyway.
    #
    # Adjusted Market Cap = Market Cap − Net Cash
    # Adjusted Earnings   = Net Income − Interest on Surplus Cash
    # Sonkin P/E          = Adjusted Market Cap / Adjusted Earnings
    net_cash    = (d["cash"] or 0) - (d["total_debt"] or 0)
    int_on_cash = net_cash * 0.04 if net_cash > 0 else 0  # assume ~4% on cash
    op_mktcap   = (v["market_cap"] or 0) - net_cash
    op_earn     = (d["net_income"] or 0) - int_on_cash
    v["sonkin_pe"] = safe_div(op_mktcap, op_earn) if op_earn and op_mktcap > 0 else None
    v["net_cash"]  = net_cash

    # ─── 1Y total return ─────────────────────────────────────────────────────
    v["total_return"] = None
    if d["price_history"] is not None and len(d["price_history"]) > 1:
        p0 = d["price_history"].iloc[0]
        p1 = d["price_history"].iloc[-1]
        v["total_return"] = safe_div(p1 - p0 + (d["dividend_ttm"] or 0), p0)

    return v


# ── Signal helpers ────────────────────────────────────────────────────────────

def mos_signal(mos_pct):
    """
    Graham's margin of safety thresholds (Ch. 3):
    ≥ 33% → adequate protection (BUY)
    10–33% → some buffer (HOLD)
    < 10% → no meaningful safety margin (EXPENSIVE)
    """
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
        "Alpha Vantage API Key",
        type="password",
        help="Free key at alphavantage.co/support/#api-key"
    )
    if not api_key:
        st.info("Get a free API key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("#### Assumptions")
    st.caption(
        "**R (Cost of Capital):** The minimum return you require. "
        "The book uses 8–12% depending on the risk profile of the business. "
        "Ch. 7: 'We think 12% is a reasonable number — several percentages "
        "higher than the long-term return on the S&P 500.'"
    )
    r = st.slider("Cost of Capital (R)", 0.04, 0.20, 0.09, 0.005, format="%.3f")

    st.caption(
        "**G (Perpetual Growth Rate):** The long-run sustainable growth rate. "
        "Ch. 7 warns: 'if G exceeds R, the value of the firm is infinite — "
        "so R is the realistic limit.' Use conservatively (2–4% typical)."
    )
    g = st.slider("Perpetual Growth Rate (G)", 0.00, 0.08, 0.03, 0.005, format="%.3f")

    st.caption(
        "**LIFO Reserve:** If the company uses LIFO inventory accounting, "
        "add the reserve disclosed in the 10-K footnotes. Ch. 4 says this "
        "converts inventory to replacement cost — a key asset adjustment."
    )
    lifo_reserve = st.number_input(
        "LIFO Reserve ($M)", min_value=0.0, value=0.0, step=10.0, format="%.1f"
    ) * 1e6

    st.divider()
    st.caption("Free tier: 25 requests/day · 5/min")
    run = st.button("🔍 Analyse", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 Value Investing Toolkit")
st.caption("Based on *Value Investing: From Graham to Buffett and Beyond* (Greenwald et al.)")

if g >= r:
    st.error("⚠️ Growth rate G must be less than Cost of Capital R. (Ch. 7: if G ≥ R, value becomes infinite — which is impossible in practice.)")
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
    "📈 Market Overview", "💰 Valuation (EPV)", "🚀 Growth Analysis", "🏭 Asset & Franchise", "🎯 Summary"
])

# ══ TAB 1 — MARKET OVERVIEW ════════════════════════════════════════════════════
with tab1:
    st.markdown(book_note(
        "Ch. 2: Before valuing a company, understand its industry and market position. "
        "The market overview provides context but is NOT the valuation — it shows what "
        "other investors are paying, which may be wrong. The value investor's job is to "
        "determine intrinsic value independently and then compare to the market price."
    ), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enterprise Value",  fmt_currency(v["ev"]),
              help="EV = Market Cap + Debt − Cash (Ch. 4)")
    c2.metric("P/E Ratio",         fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A",
              help="Market price ÷ EPS. A high P/E implies the market expects growth.")
    c3.metric("P/B Ratio",         fmt_x(d["pb_ratio"]) if d["pb_ratio"] else "N/A",
              help="Price ÷ Book Value per share. If P/B > 1, the market values the franchise above asset value.")
    c4.metric("P/S Ratio",         fmt_x(d["ps_ratio"]) if d["ps_ratio"] else "N/A")

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Beta",             f"{d['beta']:.2f}" if d["beta"] else "N/A",
               help="Market-defined risk measure. Greenwald is sceptical — he prefers MoS as the real risk measure.")
    c2b.metric("Dividend Yield",   fmt_pct(d["dividend_yield"]) if d["dividend_yield"] else "0.0%")
    c3b.metric("1Y Total Return",  fmt_pct(v["total_return"]) if v["total_return"] else "N/A")
    c4b.metric("EPS (TTM)",        f"${d['eps']:,.2f}" if d["eps"] else "N/A")

    st.markdown('<div class="section-header">Price Chart (1 Year)</div>', unsafe_allow_html=True)
    if d["price_history"] is not None:
        st.line_chart(d["price_history"], use_container_width=True)
    else:
        st.info("Price history unavailable.")

    st.markdown('<div class="section-header">Profitability</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 6: Stable, high operating margins over multiple years are strong evidence of "
        "a franchise. WD-40's consistent 24–27% EBIT margin despite zero R&D moat was "
        "the key signal. Look for margin stability, not just level."
    ), unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Gross Margin",      fmt_pct(d["gross_margin"]))
    pc2.metric("Operating Margin",  fmt_pct(d["operating_margin"]),
               help="EBIT ÷ Revenue — the book's preferred profitability measure")
    pc3.metric("Net Margin",        fmt_pct(d["net_margin"]))
    pc4.metric("ROE",               fmt_pct(d["roe"]))

# ══ TAB 2 — VALUATION (EPV) ════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Earnings Power Value (EPV) — Greenwald Core Formula</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 3 & 5: EPV = Adjusted Earnings ÷ R. "
        "This is a ZERO-GROWTH valuation — the most conservative earnings-based estimate. "
        "We start with EBIT (not net income) to remove capital-structure distortions, "
        "apply the actual tax rate to get NOPAT, add back 25% of D&A as a conservative "
        "maintenance capex buffer (Ch. 7 Intel method), then subtract debt and add surplus "
        "cash to arrive at equity EPV. 'The goal is to arrive at an accurate estimate of "
        "the current distributable cash flow.' — Ch. 3"
    ), unsafe_allow_html=True)

    # Show the EPV build-up transparently
    with st.expander("📐 EPV Calculation Breakdown", expanded=True):
        eb1, eb2, eb3 = st.columns(3)
        eb1.metric("EBIT (Operating Income)", fmt_currency(d["ebit"]),
                   help="Ch. 6: Start with EBIT, not net income — it excludes interest which is a capital-structure choice.")
        eb2.metric(f"Tax Rate ({fmt_pct(d['tax_rate'])})", "",
                   help="Effective tax rate from income statement (capped 10–45%)")
        eb3.metric("NOPAT = EBIT × (1 − tax)", fmt_currency(d.get("nopat")),
                   help="Net Operating Profit After Tax — the true operating earnings stripped of debt effects")

        eb4, eb5, eb6 = st.columns(3)
        da_add = (d["da"] or 0) * 0.25
        eb4.metric("D&A Addback (25%)", fmt_currency(da_add),
                   help="Ch. 7 Intel: 'add back 25% of D&A, assuming the other 75% will more than cover maintenance capex'")
        eb5.metric("Surplus Cash Added", fmt_currency(d.get("surplus_cash")),
                   help="Ch. 7: Cash > 1% of sales is surplus. Added to EPV since it belongs to equity holders.")
        eb6.metric("Debt Subtracted", fmt_currency(d["total_debt"]),
                   help="Ch. 7: 'subtract the book value of the interest-bearing debt' to convert enterprise EPV to equity EPV")

        st.caption(
            f"**Formula:** EPV (equity) = (NOPAT + 25% D&A) ÷ R  −  Debt  +  Surplus Cash  "
            f"=  ({fmt_currency(d.get('nopat'))} + {fmt_currency(da_add)}) ÷ {fmt_pct(r)}  "
            f"−  {fmt_currency(d['total_debt'])}  +  {fmt_currency(d.get('surplus_cash'))}"
        )

    e1, e2, e3 = st.columns(3)
    e1.metric("EPV (Total Equity)",  fmt_currency(v["epv_total"]))
    e2.metric("EPV per Share",       f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    e3.metric("Current Price",       f"${d['price']:,.2f}" if d["price"] else "N/A")

    st.markdown('<div class="section-header">Margin of Safety</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 3 & Graham: MoS = (EPV − Price) / EPV. "
        "A positive MoS means the stock trades below its zero-growth intrinsic value — "
        "you're buying a dollar for less than a dollar. Graham's threshold: ≥33% for adequate "
        "protection, ≥50% for a strong buy. A negative MoS means the market is pricing in "
        "future growth — only justifiable if you can confirm a durable franchise (see Tab 4)."
    ), unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("MoS per Share",  f"${v['mos_per_share']:,.2f}" if v["mos_per_share"] else "N/A",
              delta=fmt_pct(v["mos_pct"]) if v["mos_pct"] else None)
    m2.metric("MoS Total",      fmt_currency(v["mos_total"]))
    m3.metric("MoS %",          fmt_pct(v["mos_pct"]))
    st.markdown("**Signal:** " + mos_signal(v["mos_pct"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dividend Discount Model (DDM) — Ch. 6</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "V = Dividend ÷ (R − G). The book applies this as a cross-check for "
        "dividend-paying franchises like WD-40. It assumes dividends grow at G forever. "
        "Only reliable when dividends are stable and represent true distributable cash flow."
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

# ══ TAB 3 — GROWTH ANALYSIS ════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Growth Factor F & PV with Growth — Ch. 7</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7 Appendix: PV = C × (ROC − G) / (R − G)  where C = Invested Capital. "
        "F = (ROC − G) / (R − G) is the 'Growth Factor'. "
        "F < 1 → growth destroys value (ROC < R). "
        "F = 1 → growth is neutral (ROC = R). "
        "F > 1 → growth creates value (ROC > R). "
        "Key insight: 'Growth only adds value when return on capital exceeds cost of capital. "
        "For most companies in a competitive economy, growth adds zero value.' — Ch. 7"
    ), unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROIC",              fmt_pct(d["roic"]),
              help="NOPAT ÷ (Equity + Debt − Surplus Cash). Book formula from Ch. 7.")
    g2.metric("ROE",               fmt_pct(d["roe"]),
              help="Net Income ÷ Book Equity. Less precise than ROIC (ignores debt), used as fallback.")
    g3.metric("Growth Factor F",   fmt_x(v["growth_factor_F"]),
              help="F = (ROC − G) / (R − G). F > 1 means growth creates value.")
    g4.metric("Growth Mult. M",    fmt_x(v["growth_mult_M"]),
              help="M = PV / EPV = [1 − (G/R)(R/ROC)] / [1 − (G/R)]. Shows how much growth adds to EPV.")

    roc_used = d["roic"] or d["roe"]
    if roc_used and v["growth_factor_F"]:
        if roc_used > r:
            st.success(
                f"✅ ROC ({fmt_pct(roc_used)}) > R ({fmt_pct(r)}) — "
                f"growth creates value (F = {fmt_x(v['growth_factor_F'])}). "
                f"Only invest in the growth premium if the franchise is confirmed (Tab 4)."
            )
        elif abs(roc_used - r) < 0.01:
            st.warning("⚠️ ROC ≈ R — growth is value-neutral. Don't pay a premium for growth.")
        else:
            st.error(
                f"🔴 ROC ({fmt_pct(roc_used)}) < R ({fmt_pct(r)}) — "
                f"growth destroys value (F = {fmt_x(v['growth_factor_F'])}). "
                f"The faster this company grows, the worse for existing shareholders."
            )
    else:
        st.info("ROIC/ROE not available.")

    # PV with Growth
    st.markdown('<div class="section-header">PV with Growth vs EPV</div>', unsafe_allow_html=True)
    pv1, pv2, pv3 = st.columns(3)
    pv1.metric("EPV / Share (no growth)",     f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    pv2.metric("PV / Share (with growth)",    f"${v['pv_per_share']:,.2f}" if v["pv_per_share"] else "N/A",
               help="PV = C × (ROC − G) / (R − G), equity-adjusted. Ch. 7 formula.")
    pv3.metric("Growth Premium",
               fmt_currency((v["pv_with_growth"] or 0) - (v["epv_total"] or 0))
               if v["pv_with_growth"] and v["epv_total"] else "N/A",
               help="The extra value attributable to profitable growth above EPV.")

    # Growth Value Matrix — book's Table 7.11 reproduced as guidance
    st.markdown('<div class="section-header">Growth Value Matrix (Book Table 7.11)</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7 Table 7.11: PV/EPV ratios for combinations of ROC/R and G/R. "
        "It takes BOTH high ROC relative to R AND high G relative to R to generate "
        "meaningful growth value. Most businesses fall in the 1.0–1.5× range."
    ), unsafe_allow_html=True)
    matrix_data = {
        "G/R":     ["25%", "50%", "75%"],
        "ROC/R=1.0": [1.00, 1.00, 1.00],
        "ROC/R=1.5": [1.11, 1.33, 2.00],
        "ROC/R=2.0": [1.17, 1.50, 2.50],
        "ROC/R=2.5": [1.20, 1.60, 2.80],
        "ROC/R=3.0": [1.22, 1.67, 3.00],
    }
    df_matrix = pd.DataFrame(matrix_data).set_index("G/R")

    # Highlight the cell closest to this company's position
    if roc_used and r > 0:
        roc_r_ratio = roc_used / r
        g_r_ratio   = g / r
        st.caption(
            f"**This company's position:** ROC/R = {roc_r_ratio:.2f}×, G/R = {g_r_ratio:.0%}  "
            f"→ Growth multiplier M ≈ {fmt_x(v['growth_mult_M'])}"
        )
    st.dataframe(df_matrix, use_container_width=True)

    st.markdown('<div class="section-header">PEG Ratio — Ch. 11 (Glenn Greenberg)</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 11: PEG = P/E ÷ Growth Rate (%). Greenberg uses this as a quick screen "
        "for whether the market's growth premium is reasonable. PEG < 1.0 suggests the "
        "stock may be undervalued relative to its growth. But always verify the franchise first."
    ), unsafe_allow_html=True)
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

# ══ TAB 4 — ASSET & FRANCHISE ══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Three-Slice Valuation Summary</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 3 Figure 3.1 — The three slices of value in ascending order of uncertainty: "
        "(1) Asset Value = Reproduction Cost = what a competitor would pay to replicate this business. "
        "(2) EPV = current earnings power, zero growth assumed. "
        "(3) PV with Growth = only if ROC > R and the franchise is durable. "
        "If EPV > Asset Value → franchise evidence. If market price > EPV → growth premium being paid."
    ), unsafe_allow_html=True)

    sv1, sv2, sv3, sv4 = st.columns(4)
    sv1.metric("① Asset Value (Repro. Cost)",  fmt_currency(v["reproduction_cost"]),
               help="Book equity + LIFO reserve. A floor valuation. Ch. 4.")
    sv2.metric("② EPV (Equity)",               fmt_currency(v["epv_total"]),
               help="NOPAT/R adjusted for debt & surplus cash. Zero growth. Ch. 3, 5.")
    sv3.metric("③ PV with Growth",             fmt_currency(v["pv_with_growth"]) if v["pv_with_growth"] else "N/A",
               help="C × (ROC−G)/(R−G) equity-adjusted. Only valid if franchise confirmed. Ch. 7.")
    sv4.metric("Market Cap",                   fmt_currency(v["market_cap"]))

    # Visual comparison
    if v["epv_total"] and v["reproduction_cost"]:
        ratio = safe_div(v["epv_total"], v["reproduction_cost"])
        if ratio and ratio > 1:
            st.success(
                f"✅ EPV ({fmt_currency(v['epv_total'])}) > Reproduction Cost ({fmt_currency(v['reproduction_cost'])}) "
                f"by {fmt_x(ratio)} — this gap is the franchise value. The company earns more than a "
                f"competitive return on its assets. Evidence of a moat."
            )
        elif ratio and ratio < 0.8:
            st.error(
                f"🔴 EPV < Reproduction Cost — two explanations per Ch. 3: "
                f"(1) management is doing a poor job, or (2) the industry is in decline. "
                f"Avoid unless you have a specific catalyst for change."
            )
        else:
            st.warning("⚠️ EPV ≈ Reproduction Cost — no clear franchise. Growth adds no value.")

    st.markdown('<div class="section-header">Franchise Value — Ch. 5</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 5: Franchise Value = EPV − Reproduction Cost. "
        "A positive franchise value proves competitive advantage: the company earns more "
        "than its assets cost to reproduce, which only happens when barriers to entry "
        "protect it from competition. This is the quantitative moat test. "
        "Ch. 5: 'Without barriers to entry, sooner or later competition will force the rate "
        "of return downward until it equals the cost of capital.'"
    ), unsafe_allow_html=True)
    fv1, fv2, fv3 = st.columns(3)
    fv1.metric("EPV (Equity)",        fmt_currency(v["epv_total"]))
    fv2.metric("Reproduction Cost",   fmt_currency(v["reproduction_cost"]),
               help="Book equity + LIFO reserve. Ch. 4: what a competitor must invest to replicate this business.")
    fv3.metric("Franchise Value",     fmt_currency(v["franchise_value"]),
               help="EPV − Reproduction Cost. Positive = moat. Negative = no competitive advantage.")

    if v["franchise_value"] and v["epv_total"]:
        fv_pct = safe_div(v["franchise_value"], v["epv_total"])
        if v["franchise_value"] > 0:
            st.success(f"✅ Franchise value = {fmt_currency(v['franchise_value'])} ({fmt_pct(fv_pct)} of EPV) — competitive moat likely.")
        else:
            st.warning("⚠️ EPV ≤ Reproduction Cost — limited evidence of a durable moat.")

    st.markdown('<div class="section-header">ROIC Components — Ch. 7</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7: ROIC = NOPAT ÷ Invested Capital. "
        "Invested Capital = Equity + Debt − Surplus Cash (cash > 1% of revenue). "
        "Surplus cash is stripped out because it earns only interest, not operating returns. "
        "A sustained ROIC > R is the quantitative signature of a franchise."
    ), unsafe_allow_html=True)
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("NOPAT",             fmt_currency(d.get("nopat")))
    rc2.metric("Invested Capital",  fmt_currency(d.get("invested_capital")))
    rc3.metric("Surplus Cash",      fmt_currency(d.get("surplus_cash")))
    rc4.metric("ROIC",              fmt_pct(d["roic"]))

    st.markdown('<div class="section-header">Cap Rate — Ch. 10 (Mario Gabelli)</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 10: Cap Rate = (EBITDA − Capex) ÷ Enterprise Value. "
        "Gabelli's 'Private Market Value' approach asks: what yield would a strategic "
        "acquirer get on this business? Inspired by real-estate cap rates. "
        "A cap rate > 8% signals the business is attractively priced."
    ), unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("EBITDA − Capex",   fmt_currency(v["operating_cf_for_cap"]))
    ca2.metric("Enterprise Value", fmt_currency(v["ev"]))
    ca3.metric("Cap Rate",         fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A")
    st.markdown("**Signal:** " + cap_signal(v["cap_rate"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sonkin Adjusted P/E — Ch. 16</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 16: Paul Sonkin's technique for small-cap stocks with significant net cash. "
        "Strips net cash out of both market cap and earnings so you only pay for the "
        "operating business. Prevents the value trap where cash inflates apparent cheapness. "
        "Adjusted P/E = (Market Cap − Net Cash) ÷ (Net Income − Interest on Surplus Cash)"
    ), unsafe_allow_html=True)
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Net Cash",         fmt_currency(v["net_cash"]), help="Cash − Total Debt")
    sp2.metric("Op. Market Cap",   fmt_currency((v["market_cap"] or 0) - v["net_cash"]))
    sp3.metric("Sonkin Adj. P/E",  fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A")

# ══ TAB 5 — SUMMARY ════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 🎯 Summary: {d['name']} [{ticker}]")
    st.caption(f"R = {fmt_pct(r)}  |  G = {fmt_pct(g)}  |  {pd.Timestamp.now().strftime('%d %b %Y')}")

    summary_rows = [
        ("Market",    "Price",                f"${d['price']:,.2f}" if d["price"] else "N/A",   "—"),
        ("Market",    "Market Cap",            fmt_currency(v["market_cap"]),                    "—"),
        ("Market",    "Enterprise Value",      fmt_currency(v["ev"]),                            "—"),
        ("Market",    "1Y Total Return",       fmt_pct(v["total_return"]) if v["total_return"] else "N/A", "—"),
        ("Valuation", "① Reproduction Cost",   fmt_currency(v["reproduction_cost"]),             "—"),
        ("Valuation", "② EPV / Share",         f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A", "—"),
        ("Valuation", "③ PV w/ Growth / Share",f"${v['pv_per_share']:,.2f}" if v["pv_per_share"] else "N/A", "—"),
        ("Valuation", "Margin of Safety",      fmt_pct(v["mos_pct"]) if v["mos_pct"] else "N/A",
                                               "✅ BUY" if (v["mos_pct"] or 0) >= 0.33
                                               else ("⚠️ HOLD" if (v["mos_pct"] or 0) >= 0.10 else "🔴 SELL")),
        ("Valuation", "DDM / Share",           f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A", "—"),
        ("Valuation", "P/E Ratio",             fmt_x(d["pe_ratio"]) if d["pe_ratio"] else "N/A", "—"),
        ("Growth",    "ROIC",                  fmt_pct(d["roic"]) if d["roic"] else "N/A",
                                               "✅ ROC>R" if (d["roic"] or 0) > r else ("⚠️ ROC≈R" if abs((d["roic"] or 0) - r) < 0.02 else "🔴 ROC<R")),
        ("Growth",    "Growth Factor F",       fmt_x(v["growth_factor_F"]) if v["growth_factor_F"] else "N/A",
                                               "✅ Value" if (v["growth_factor_F"] or 0) > 1 else "⚠️ Neutral"),
        ("Growth",    "PEG Ratio",             fmt_x(v["peg"]) if v["peg"] else "N/A",
                                               "✅ <1" if v["peg"] and v["peg"] < 1
                                               else ("⚠️ 1–3" if v["peg"] and v["peg"] < 3 else "🔴 >3")),
        ("Asset",     "Franchise Value",       fmt_currency(v["franchise_value"]),
                                               "✅ Moat" if (v["franchise_value"] or 0) > 0 else "⚠️ None"),
        ("Asset",     "Cap Rate",              fmt_pct(v["cap_rate"]) if v["cap_rate"] else "N/A",
                                               "✅ >8%" if (v["cap_rate"] or 0) >= 0.08
                                               else ("⚠️ 5–8%" if (v["cap_rate"] or 0) >= 0.05 else "🔴 <5%")),
        ("Asset",     "Sonkin Adj. P/E",       fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "N/A", "—"),
    ]

    df_summary = pd.DataFrame(summary_rows, columns=["Category", "Metric", "Value", "Signal"])

    def style_signal(val):
        if "✅" in str(val): return "color: #00e676; font-weight: 600"
        if "⚠️" in str(val): return "color: #ffd740; font-weight: 600"
        if "🔴" in str(val): return "color: #ff5252; font-weight: 600"
        return "color: #888"

    styled = df_summary.style.map(style_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

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

    st.markdown(book_note(
        "Remember the book's hierarchy: Asset Value is most reliable, EPV is second, "
        "Growth Value is least reliable. A safe investment ideally shows: "
        "Price < EPV < PV with Growth. The margin of safety is your protection "
        "against estimation errors. 'The greater the margin of safety, the lower the risk.' — Ch. 3"
    ), unsafe_allow_html=True)
    st.caption("⚠️ Research only — not financial advice. Verify data independently.")
