"""
Value Investing Toolkit — Streamlit App
Based on: Value Investing: From Graham to Buffett and Beyond (Greenwald, Kahn, Sonkin, van Biema)
Data: SEC EDGAR XBRL API (completely free, no API key, no rate limits for normal use)

WHY SEC EDGAR?
──────────────
• 100% free — no API key, no account, no payment ever
• No daily call limits (10 req/sec max, well within our 2-call design)
• Primary source — this is where Bloomberg, FMP, Alpha Vantage all get their data
• Official 10-K/10-Q filings — the most accurate numbers possible

HOW IT WORKS (2 calls per ticker):
  Call 1: sec.gov/files/company_tickers.json   → ticker → CIK number
  Call 2: data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
            → every financial fact ever filed, organised by US-GAAP concept

IMPORTANT: Only works for US-listed stocks (SEC filers). International stocks
and some ETFs/funds won't have EDGAR data.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

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
    font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #888;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #2d2d2d; padding-bottom: 6px; margin: 20px 0 12px 0;
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


# ── SEC EDGAR fetch layer ─────────────────────────────────────────────────────

SEC_HEADERS = {"User-Agent": "ValueInvestingToolkit research@valuetoolkit.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_to_cik() -> dict:
    """
    Fetch the SEC's full ticker→CIK mapping (one call, cached 1 hour).
    Returns dict of TICKER → zero-padded 10-digit CIK string.
    """
    r = requests.get(TICKERS_URL, headers=SEC_HEADERS, timeout=15)
    r.raise_for_status()
    raw = r.json()
    return {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in raw.values()
    }

@st.cache_data(ttl=28800, show_spinner=False)   # 8-hour cache per CIK
def get_company_facts(cik: str) -> dict:
    """
    Fetch all XBRL facts for a company (one call per ticker, cached 8 hours).
    Returns the full companyfacts JSON — every concept ever filed.
    """
    url = FACTS_URL.format(cik=cik)
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)   # 5-min cache — price changes often
def get_price_yf(ticker: str) -> float | None:
    """
    Fetch the current stock price from Yahoo Finance via yfinance.
    Falls back to None if yfinance is unavailable or the ticker isn't found.
    Cache TTL is 5 minutes so the price stays reasonably fresh.
    """
    if not _YF_AVAILABLE:
        return None
    try:
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        return float(price) if price else None
    except Exception:
        return None


def pick_annual(entries: list, prefer_fy: int = None) -> dict | None:
    """
    From a list of XBRL fact entries, pick the most recent 10-K annual value.
    Filters to form=10-K, fp=FY, then sorts by end date descending.
    If prefer_fy given, tries that fiscal year first.
    """
    annual = [e for e in entries
              if e.get("form") in ("10-K", "10-K/A")
              and e.get("fp") == "FY"]
    if not annual:
        return None
    annual.sort(key=lambda x: x.get("end", ""), reverse=True)
    if prefer_fy:
        fy_match = [e for e in annual if e.get("fy") == prefer_fy]
        if fy_match:
            return fy_match[0]
    return annual[0]

def pick_prior_annual(entries: list, latest_fy: int) -> dict | None:
    """Pick the 10-K entry for the fiscal year before latest_fy."""
    annual = [e for e in entries
              if e.get("form") in ("10-K", "10-K/A")
              and e.get("fp") == "FY"
              and e.get("fy") == latest_fy - 1]
    if not annual:
        return None
    annual.sort(key=lambda x: x.get("end", ""), reverse=True)
    return annual[0]

def get_concept(facts_usgaap: dict, *concept_names) -> tuple[list, str]:
    """
    Try multiple US-GAAP concept names in order, return first hit.
    Companies use slightly different tags for the same line item.
    Returns (entries_list, concept_name_used).
    """
    for name in concept_names:
        entries = facts_usgaap.get(name, {}).get("units", {}).get("USD", [])
        if entries:
            return entries, name
    return [], concept_names[0]

def val(entry: dict | None) -> float | None:
    """Extract float value from a fact entry, or None."""
    if entry is None:
        return None
    return safe_float(entry.get("val"))


# ── Map EDGAR concepts → financial dict ──────────────────────────────────────

def extract_financials(facts: dict, ticker: str) -> dict:
    """
    Parse the companyfacts JSON into the clean dict used by compute_valuations().

    EDGAR XBRL concept mapping — companies sometimes use different tags:
      Revenue:     Revenues | RevenueFromContractWithCustomerExcludingAssessedTax
                   | SalesRevenueNet | RevenueFromContractWithCustomer
      EBIT:        OperatingIncomeLoss
      Net income:  NetIncomeLoss | ProfitLoss
      Tax:         IncomeTaxExpense | IncomeTaxesPaid
      Pretax:      IncomeLossFromContinuingOperationsBeforeIncomeTaxes...
      D&A:         DepreciationDepletionAndAmortization | DepreciationAndAmortization
      Capex:       PaymentsToAcquirePropertyPlantAndEquipment
      Cash:        CashAndCashEquivalentsAtCarryingValue | Cash
                   | CashCashEquivalentsAndShortTermInvestments
      Total debt:  LongTermDebtAndCapitalLeaseObligation + ShortTermBorrowings
                   | DebtCurrent + LongTermDebt
      Equity:      StockholdersEquity | StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
      Shares:      CommonStockSharesOutstanding | EntityCommonStockSharesOutstanding
      Op CF:       NetCashProvidedByUsedInOperatingActivities
      Dividends:   PaymentsOfDividends | PaymentsOfDividendsCommonStock
    """
    d = {}
    d["name"]   = facts.get("entityName", ticker)
    d["ticker"] = ticker

    usgaap = facts.get("facts", {}).get("us-gaap", {})
    dei    = facts.get("facts", {}).get("dei", {})

    # ── Revenue ──────────────────────────────────────────────────────────────
    rev_entries, _ = get_concept(usgaap,
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    )
    rev_latest = pick_annual(rev_entries)
    rev_prior  = pick_prior_annual(rev_entries, rev_latest["fy"]) if rev_latest else None
    latest_fy  = rev_latest["fy"] if rev_latest else None

    d["revenue"]       = val(rev_latest)
    d["revenue_prior"] = val(rev_prior)
    d["fiscal_year"]   = latest_fy

    # ── EBIT (Operating Income) ───────────────────────────────────────────────
    ebit_entries, _ = get_concept(usgaap, "OperatingIncomeLoss")
    d["ebit"] = val(pick_annual(ebit_entries, latest_fy))

    # ── Net Income ───────────────────────────────────────────────────────────
    ni_entries, _ = get_concept(usgaap,
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    )
    ni_latest = pick_annual(ni_entries, latest_fy)
    ni_prior  = pick_prior_annual(ni_entries, latest_fy) if latest_fy else None
    d["net_income"]       = val(ni_latest)
    d["net_income_prior"] = val(ni_prior)

    # ── Tax rate ─────────────────────────────────────────────────────────────
    tax_entries, _ = get_concept(usgaap, "IncomeTaxExpenseBenefit")
    pretax_entries, _ = get_concept(usgaap,
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    )
    income_tax = val(pick_annual(tax_entries, latest_fy)) or 0
    pretax     = val(pick_annual(pretax_entries, latest_fy)) or 0
    if pretax > 0 and income_tax:
        d["tax_rate"] = min(max(income_tax / pretax, 0.10), 0.45)
    else:
        d["tax_rate"] = 0.21

    # ── EBITDA (derived: EBIT + D&A) ─────────────────────────────────────────
    da_entries, _ = get_concept(usgaap,
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    )
    d["da"] = abs(val(pick_annual(da_entries, latest_fy)) or 0)
    d["ebitda"] = (d["ebit"] or 0) + d["da"] if d["ebit"] else None

    # ── Gross Profit ─────────────────────────────────────────────────────────
    gp_entries, _ = get_concept(usgaap, "GrossProfit")
    d["gross_profit"] = val(pick_annual(gp_entries, latest_fy))

    # ── EPS ──────────────────────────────────────────────────────────────────
    eps_entries, _ = get_concept(usgaap,
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    )
    # EPS is in USD/shares, not USD — XBRL units key differs
    eps_usd = usgaap.get("EarningsPerShareDiluted", {}).get("units", {}).get("USD/shares", []) or \
              usgaap.get("EarningsPerShareBasic",   {}).get("units", {}).get("USD/shares", [])
    eps_ann = [e for e in eps_usd if e.get("form") in ("10-K","10-K/A") and e.get("fp")=="FY"]
    eps_ann.sort(key=lambda x: x.get("end",""), reverse=True)
    d["eps"] = val(eps_ann[0]) if eps_ann else safe_div(d["net_income"], None)

    # ── Shares outstanding ───────────────────────────────────────────────────
    sh_entries = usgaap.get("CommonStockSharesOutstanding", {}).get("units", {}).get("shares", []) or \
                 dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    sh_ann = [e for e in sh_entries if e.get("form") in ("10-K","10-K/A","DEF 14A")]
    sh_ann.sort(key=lambda x: x.get("end",""), reverse=True)
    d["shares"] = val(sh_ann[0]) if sh_ann else None

    # ── Balance sheet — Cash ─────────────────────────────────────────────────
    cash_entries, _ = get_concept(usgaap,
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    )
    d["cash"] = val(pick_annual(cash_entries, latest_fy))

    # ── Balance sheet — Total Debt ───────────────────────────────────────────
    # Try combined debt first, fall back to long + short
    ltd_entries, _ = get_concept(usgaap,
        "LongTermDebtAndCapitalLeaseObligation",
        "LongTermDebt",
    )
    std_entries, _ = get_concept(usgaap,
        "ShortTermBorrowings",
        "DebtCurrent",
        "CommercialPaper",
    )
    ltd = val(pick_annual(ltd_entries, latest_fy)) or 0
    std = val(pick_annual(std_entries, latest_fy)) or 0
    d["total_debt"] = ltd + std if (ltd or std) else None

    # ── Balance sheet — Equity ───────────────────────────────────────────────
    eq_entries, _ = get_concept(usgaap,
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
    )
    d["total_equity"] = val(pick_annual(eq_entries, latest_fy))
    d["book_value_ps"] = safe_div(d["total_equity"], d["shares"])

    # ── Cash Flow — Capex ────────────────────────────────────────────────────
    capex_entries, _ = get_concept(usgaap,
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    )
    d["capex"] = abs(val(pick_annual(capex_entries, latest_fy)) or 0)

    # ── Cash Flow — Operating ────────────────────────────────────────────────
    ocf_entries, _ = get_concept(usgaap,
        "NetCashProvidedByUsedInOperatingActivities",
    )
    d["operating_cf"] = val(pick_annual(ocf_entries, latest_fy))

    # ── Cash Flow — Dividends ────────────────────────────────────────────────
    div_entries, _ = get_concept(usgaap,
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfOrdinaryDividends",
    )
    d["dividends_paid"] = abs(val(pick_annual(div_entries, latest_fy)) or 0)
    d["dividend_ttm"] = safe_div(d["dividends_paid"], d["shares"]) \
                        if d["shares"] and d["dividends_paid"] else 0

    # ── Market data — price and market cap not in EDGAR XBRL ─────────────────
    # EDGAR only has fundamental filing data, not real-time prices.
    # We derive what we can from the filings; price must come from elsewhere.
    # For now we set these to None and tell the user to enter price manually.
    d["price"]      = None
    d["market_cap"] = None
    d["beta"]       = None

    # ── Sector / industry — from EDGAR entity metadata ────────────────────────
    # EDGAR doesn't tag sector in companyfacts; use SIC description if available
    d["sector"]   = facts.get("facts", {}).get("dei", {}).get("SicDescription", {})
    d["industry"] = "N/A"

    # ── Growth ───────────────────────────────────────────────────────────────
    if d["revenue"] and d["revenue_prior"] and d["revenue_prior"] != 0:
        d["revenue_growth"] = (d["revenue"] - d["revenue_prior"]) / abs(d["revenue_prior"])
    else:
        d["revenue_growth"] = None

    if d["net_income"] and d["net_income_prior"] and d["net_income_prior"] != 0:
        d["earnings_growth"] = (d["net_income"] - d["net_income_prior"]) / abs(d["net_income_prior"])
    else:
        d["earnings_growth"] = None

    # ── Profitability ratios ──────────────────────────────────────────────────
    d["gross_margin"]     = safe_div(d["gross_profit"], d["revenue"])
    d["operating_margin"] = safe_div(d["ebit"], d["revenue"])
    d["net_margin"]       = safe_div(d["net_income"], d["revenue"])
    d["roe"]              = safe_div(d["net_income"], d["total_equity"])

    # ── ROIC — book-correct formula (Ch. 7) ──────────────────────────────────
    # NOPAT = EBIT × (1 − tax_rate)
    # Surplus Cash = Cash − 1% of Revenue  (Ch. 7: "1% of sales" standard)
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

    # ── EV (without price we can only do debt + equity book) ─────────────────
    d["ev_dollars"] = None   # requires market cap (price × shares)

    d["price_history"] = None
    return d


# ── Valuations ────────────────────────────────────────────────────────────────

def compute_valuations(d, r, g, lifo_reserve=0.0, manual_price=None):
    v = {}
    rg = r - g

    # Use manually-entered price if provided
    price = manual_price or d.get("price")
    d["price"] = price

    shares = d.get("shares")
    market_cap = (price * shares) if price and shares else None
    d["market_cap"] = market_cap
    v["market_cap"] = market_cap

    total_debt = d.get("total_debt") or 0
    cash       = d.get("cash") or 0
    v["ev"]    = (market_cap + total_debt - cash) if market_cap else None

    # ── Asset Value / Reproduction Cost (Ch. 4) ──────────────────────────────
    book_total = (d.get("book_value_ps") or 0) * (shares or 0) \
                 if d.get("book_value_ps") and shares \
                 else (d.get("total_equity") or 0)
    v["reproduction_cost"] = book_total + lifo_reserve

    # ── EPV (Ch. 3, 5, 6, 7) ─────────────────────────────────────────────────
    # NOPAT + 25% D&A addback ÷ R → enterprise EPV → subtract debt, add surplus cash
    if d.get("ebit"):
        nopat_epv  = d["ebit"] * (1 - d["tax_rate"])
        da_addback = (d.get("da") or 0) * 0.25
        adj_earn   = nopat_epv + da_addback
    else:
        adj_earn = d.get("net_income") or ((d.get("eps") or 0) * (shares or 1))

    surplus = d.get("surplus_cash") or 0
    epv_ent         = safe_div(adj_earn, r)
    v["epv_total"]  = (epv_ent - total_debt + surplus) if epv_ent is not None else None
    v["epv_per_share"] = safe_div(v["epv_total"], shares)
    v["adj_earnings"]  = adj_earn

    # ── Margin of Safety ──────────────────────────────────────────────────────
    v["mos_per_share"] = ((v["epv_per_share"] or 0) - (price or 0)) \
                         if v["epv_per_share"] and price else None
    v["mos_total"]     = (v["epv_total"] or 0) - (market_cap or 0) \
                         if v["epv_total"] and market_cap else None
    v["mos_pct"]       = safe_div(v["mos_per_share"], v["epv_per_share"]) \
                         if v["epv_per_share"] and v["epv_per_share"] != 0 else None

    # ── Franchise Value = EPV − Reproduction Cost (Ch. 5) ────────────────────
    v["franchise_value"]     = (v["epv_total"] or 0) - v["reproduction_cost"] \
                               if v["epv_total"] else None
    v["franchise_per_share"] = safe_div(v["franchise_value"], shares)

    # ── PV with Growth = C × (ROC − G) / (R − G)  (Ch. 7) ───────────────────
    roc = d.get("roic") or d.get("roe") or 0
    v["growth_factor_F"] = safe_div(roc - g, rg) if rg > 0 and roc else None

    inv_cap = d.get("invested_capital") or 0
    if rg > 0 and roc and inv_cap > 0:
        pv_ent              = inv_cap * safe_div(roc - g, rg)
        v["pv_with_growth"] = (pv_ent - total_debt + surplus) if pv_ent else None
        v["pv_per_share"]   = safe_div(v["pv_with_growth"], shares)
    else:
        v["pv_with_growth"] = v["pv_per_share"] = None

    # Growth Multiplier M (Ch. 7)
    if r > 0 and roc and roc > 0:
        gr    = g / r
        denom = 1 - gr * (r / roc)
        v["growth_mult_M"] = safe_div(1 - gr, denom) if denom != 0 else None
    else:
        v["growth_mult_M"] = None

    # ── DDM  V = D / (R − G) (Ch. 6) ─────────────────────────────────────────
    div = d.get("dividend_ttm") or 0
    if rg > 0 and div > 0:
        v["ddm_per_share"] = div / rg
        v["ddm_total"]     = v["ddm_per_share"] * (shares or 0)
    else:
        v["ddm_per_share"] = v["ddm_total"] = None

    # ── Cap Rate  (EBITDA − Capex) / EV  (Ch. 10) ────────────────────────────
    op_cf = (d.get("ebitda") or 0) - (d.get("capex") or 0)
    v["cap_rate"]             = safe_div(op_cf, v["ev"]) if v["ev"] else None
    v["operating_cf_for_cap"] = op_cf

    # ── PEG (Ch. 11) ──────────────────────────────────────────────────────────
    # Store in v[] not d[] — d is cached, v is recomputed every run.
    # This ensures ratios always reflect the current manually-entered price.
    v["pe_ratio"] = safe_div(price, d.get("eps")) if price and d.get("eps") else None
    g_pct = (d.get("earnings_growth") or d.get("revenue_growth") or g) * 100
    v["peg"] = safe_div(v["pe_ratio"], g_pct) if v["pe_ratio"] and g_pct else None

    # ── P/B and P/S ───────────────────────────────────────────────────────────
    v["pb_ratio"] = safe_div(price, d.get("book_value_ps")) \
                    if price and d.get("book_value_ps") else None
    rev_ps = safe_div(d.get("revenue"), shares)
    v["ps_ratio"] = safe_div(price, rev_ps) if price and rev_ps else None

    # ── Sonkin Adjusted P/E  (Ch. 16) ─────────────────────────────────────────
    net_cash    = cash - total_debt
    int_on_cash = net_cash * 0.04 if net_cash > 0 else 0
    op_mktcap   = (market_cap or 0) - net_cash
    op_earn     = (d.get("net_income") or 0) - int_on_cash
    v["sonkin_pe"] = safe_div(op_mktcap, op_earn) if op_earn and op_mktcap > 0 else None
    v["net_cash"]  = net_cash

    v["total_return"] = None
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
    st.caption("Greenwald · Graham · Hooke  ·  Data: SEC EDGAR + Yahoo Finance")
    st.divider()

    st.markdown("**No API key needed** — fundamentals from SEC 10-K · price from Yahoo Finance.")
    st.caption("US stocks only (SEC filers). International stocks not supported.")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("#### 💵 Current Stock Price")
    # Auto-fetch from Yahoo Finance; user can override manually
    _auto_price = get_price_yf(ticker_input) if ticker_input else None
    _price_default = round(float(_auto_price), 2) if _auto_price else 0.0
    manual_price = st.number_input(
        "Current price ($)",
        min_value=0.0, value=_price_default, step=0.01, format="%.2f",
        help="Auto-fetched from Yahoo Finance. You can edit this to use a different price."
    )
    manual_price = manual_price if manual_price > 0 else None
    if _auto_price:
        st.caption(f"📡 Yahoo Finance live price: ${_auto_price:,.2f}")
    else:
        st.caption("⚠️ Could not fetch price — enter manually.")

    st.markdown("#### Assumptions")
    r = st.slider("Cost of Capital (R)", 0.04, 0.20, 0.09, 0.005, format="%.3f",
                  help="Ch. 7: '12% is reasonable — above the long-term S&P 500 return.'")
    g = st.slider("Perpetual Growth Rate (G)", 0.00, 0.08, 0.03, 0.005, format="%.3f",
                  help="Ch. 7: 'If G ≥ R, value becomes infinite — R is the ceiling.'")
    lifo_reserve = st.number_input(
        "LIFO Reserve ($M)", min_value=0.0, value=0.0, step=10.0, format="%.1f",
        help="Ch. 4: From 10-K footnotes. Converts LIFO inventory to replacement cost."
    ) * 1e6

    st.divider()
    run = st.button("🔍 Analyse", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 Value Investing Toolkit")
st.caption("Based on *Value Investing: From Graham to Buffett and Beyond* (Greenwald et al.)  ·  Fundamentals: SEC EDGAR · Price: Yahoo Finance")

if g >= r:
    st.error("⚠️ G must be < R. (Ch. 7: if G ≥ R, value becomes infinite.)")
    st.stop()

if not run and "last_ticker" not in st.session_state:
    st.info("Enter a US stock ticker in the sidebar and click **Analyse**.")
    st.stop()

ticker = ticker_input if run else st.session_state.get("last_ticker", ticker_input)
if run:
    st.session_state["last_ticker"] = ticker_input

# ── Fetch CIK then CompanyFacts ───────────────────────────────────────────────
with st.spinner(f"Looking up {ticker} on SEC EDGAR…"):
    try:
        ticker_map = get_ticker_to_cik()
        cik = ticker_map.get(ticker.upper())
        if not cik:
            st.error(f"**{ticker}** not found in SEC EDGAR. Only US-listed stocks are supported.")
            st.stop()
    except Exception as e:
        st.error(f"Could not reach SEC EDGAR: {e}")
        st.stop()

with st.spinner(f"Fetching 10-K data for {ticker} (CIK {cik})…"):
    try:
        facts = get_company_facts(cik)
    except Exception as e:
        st.error(f"EDGAR data error for {ticker}: {e}")
        st.stop()

d = extract_financials(facts, ticker)
v = compute_valuations(d, r, g, lifo_reserve, manual_price)

# ── Header ─────────────────────────────────────────────────────────────────────
ca, cb, cc, cd = st.columns([3, 1, 1, 1])
with ca:
    st.subheader(f"{d['name']}  [{ticker}]")
    fy_label = f"FY{d['fiscal_year']}" if d.get("fiscal_year") else "FY data"
    st.caption(f"SEC EDGAR · {fy_label} 10-K · CIK {cik}")
cb.metric("Price",      f"${manual_price:,.2f}" if manual_price else "Enter →")
cc.metric("Market Cap", fmt_currency(v["market_cap"]))
cd.metric("Fiscal Year", str(d.get("fiscal_year", "N/A")))

if not manual_price:
    st.info("💡 Price could not be fetched automatically. Enter it in the sidebar to unlock all price-based metrics.")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Financials", "💰 Valuation (EPV)", "🚀 Growth Analysis", "🏭 Asset & Franchise", "🎯 Summary"
])

# ══ TAB 1 — FINANCIALS ════════════════════════════════════════════════════════
with tab1:
    st.markdown(book_note(
        "Data sourced directly from SEC 10-K annual filings via EDGAR XBRL. "
        "This is the same primary source used by Bloomberg, FMP, and Alpha Vantage — "
        "but accessed for free, directly."
    ), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue",       fmt_currency(d["revenue"]))
    c2.metric("EBIT",          fmt_currency(d["ebit"]),
              help="Operating Income from 10-K — Ch. 6/7 preferred EPV starting point")
    c3.metric("Net Income",    fmt_currency(d["net_income"]))
    c4.metric("EBITDA",        fmt_currency(d["ebitda"]))

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Total Debt",   fmt_currency(d["total_debt"]))
    c2b.metric("Cash",         fmt_currency(d["cash"]))
    c3b.metric("Total Equity", fmt_currency(d["total_equity"]))
    c4b.metric("Capex",        fmt_currency(d["capex"]))

    st.markdown('<div class="section-header">Market Multiples (requires price)</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("P/E Ratio",  fmt_x(v["pe_ratio"])  if v["pe_ratio"]  else "Enter price →")
    mc2.metric("P/B Ratio",  fmt_x(v["pb_ratio"])  if v["pb_ratio"]  else "Enter price →")
    mc3.metric("P/S Ratio",  fmt_x(v["ps_ratio"])  if v["ps_ratio"]  else "Enter price →")
    mc4.metric("EPS (Dil.)", f"${d['eps']:,.2f}"   if d.get("eps")   else "N/A")

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

# ══ TAB 2 — EPV ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Earnings Power Value (EPV) — Greenwald</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 3: EPV = Adjusted Earnings ÷ R. Zero-growth, most conservative. "
        "Ch. 6/7: Start with EBIT, apply tax → NOPAT, add back 25% D&A (maintenance capex buffer), "
        "subtract debt, add surplus cash → equity EPV."
    ), unsafe_allow_html=True)

    with st.expander("📐 EPV Build-up", expanded=True):
        eb1, eb2, eb3 = st.columns(3)
        eb1.metric("EBIT (from 10-K)", fmt_currency(d["ebit"]))
        eb2.metric("Tax Rate",         fmt_pct(d["tax_rate"]))
        eb3.metric("NOPAT",            fmt_currency(d.get("nopat")))
        eb4, eb5, eb6 = st.columns(3)
        da_add = (d.get("da") or 0) * 0.25
        eb4.metric("D&A Addback (25%)", fmt_currency(da_add),
                   help="Ch. 7 Intel: 25% of D&A as conservative maintenance capex buffer")
        eb5.metric("Surplus Cash",      fmt_currency(d.get("surplus_cash")),
                   help="Ch. 7: Cash > 1% of sales. Added to equity EPV.")
        eb6.metric("Debt Subtracted",   fmt_currency(d.get("total_debt")),
                   help="Ch. 7: Subtract debt to convert enterprise EPV to equity EPV.")
        st.caption("EPV (equity) = (NOPAT + 25% D&A) ÷ R  −  Debt  +  Surplus Cash")

    e1, e2, e3 = st.columns(3)
    e1.metric("EPV (Total Equity)", fmt_currency(v["epv_total"]))
    e2.metric("EPV per Share",      f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    e3.metric("Current Price",      f"${manual_price:,.2f}" if manual_price else "Enter in sidebar →")

    st.markdown('<div class="section-header">Margin of Safety</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Graham / Ch. 3: MoS = (EPV − Price) / EPV. "
        "≥33% = adequate protection. ≥50% = strong buy. "
        "Negative MoS = market pricing in growth — only justified with confirmed franchise."
    ), unsafe_allow_html=True)
    if manual_price:
        m1, m2, m3 = st.columns(3)
        m1.metric("MoS per Share", f"${v['mos_per_share']:,.2f}" if v["mos_per_share"] else "N/A",
                  delta=fmt_pct(v["mos_pct"]) if v["mos_pct"] else None)
        m2.metric("MoS Total",     fmt_currency(v["mos_total"]))
        m3.metric("MoS %",         fmt_pct(v["mos_pct"]))
        st.markdown("**Signal:** " + mos_signal(v["mos_pct"]), unsafe_allow_html=True)
    else:
        st.info("Enter stock price in sidebar to calculate Margin of Safety.")

    st.markdown('<div class="section-header">Dividend Discount Model (DDM) — Ch. 6</div>', unsafe_allow_html=True)
    if d.get("dividend_ttm") and d["dividend_ttm"] > 0:
        d1, d2, d3 = st.columns(3)
        d1.metric("Dividend / Share", f"${d['dividend_ttm']:,.4f}")
        d2.metric("DDM Value",        f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A")
        d3.metric("DDM vs Price",
                  f"${(v['ddm_per_share'] or 0) - (manual_price or 0):+,.2f}"
                  if v["ddm_per_share"] and manual_price else "N/A")
    else:
        st.warning("No dividends paid — DDM not applicable.")

# ══ TAB 3 — GROWTH ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Growth Factor F & PV with Growth — Ch. 7</div>', unsafe_allow_html=True)
    st.markdown(book_note(
        "Ch. 7: PV = C × (ROC − G) / (R − G). "
        "F = (ROC − G) / (R − G). F > 1 = growth creates value. "
        "'Growth only adds value when ROC > R.' — Ch. 7"
    ), unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROIC",            fmt_pct(d.get("roic")),        help="NOPAT ÷ Invested Capital (Ch. 7 formula)")
    g2.metric("ROE",             fmt_pct(d.get("roe")),          help="Fallback when ROIC unavailable")
    g3.metric("Growth Factor F", fmt_x(v["growth_factor_F"]),   help="F = (ROC−G)/(R−G)")
    g4.metric("Growth Mult. M",  fmt_x(v["growth_mult_M"]),     help="M = PV/EPV")

    roc_used = d.get("roic") or d.get("roe")
    if roc_used and v["growth_factor_F"]:
        if roc_used > r:
            st.success(f"✅ ROC ({fmt_pct(roc_used)}) > R ({fmt_pct(r)}) — growth creates value (F = {fmt_x(v['growth_factor_F'])}).")
        elif abs(roc_used - r) < 0.01:
            st.warning("⚠️ ROC ≈ R — growth is value-neutral.")
        else:
            st.error(f"🔴 ROC ({fmt_pct(roc_used)}) < R ({fmt_pct(r)}) — growth destroys value.")

    st.markdown('<div class="section-header">PV with Growth vs EPV</div>', unsafe_allow_html=True)
    pv1, pv2, pv3 = st.columns(3)
    pv1.metric("EPV / Share",           f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A")
    pv2.metric("PV / Share (w/ growth)", f"${v['pv_per_share']:,.2f}"  if v["pv_per_share"]  else "N/A")
    pv3.metric("Growth Premium",
               fmt_currency((v["pv_with_growth"] or 0) - (v["epv_total"] or 0))
               if v["pv_with_growth"] and v["epv_total"] else "N/A")

    st.markdown('<div class="section-header">Growth Value Matrix — Book Table 7.11</div>', unsafe_allow_html=True)
    matrix_df = pd.DataFrame({
        "G/R":       ["25%", "50%", "75%"],
        "ROC/R=1.0": [1.00, 1.00, 1.00],
        "ROC/R=1.5": [1.11, 1.33, 2.00],
        "ROC/R=2.0": [1.17, 1.50, 2.50],
        "ROC/R=2.5": [1.20, 1.60, 2.80],
        "ROC/R=3.0": [1.22, 1.67, 3.00],
    }).set_index("G/R")
    if roc_used and r > 0:
        st.caption(f"This company: ROC/R = {roc_used/r:.2f}×, G/R = {g/r:.0%} → M ≈ {fmt_x(v['growth_mult_M'])}")
    st.dataframe(matrix_df, use_container_width=True)

    st.markdown('<div class="section-header">Revenue & Earnings Trends</div>', unsafe_allow_html=True)
    tr1, tr2, tr3 = st.columns(3)
    tr1.metric("Revenue Growth (YOY)",  fmt_pct(d.get("revenue_growth")))
    tr2.metric("Earnings Growth (YOY)", fmt_pct(d.get("earnings_growth")))
    tr3.metric("Revenue (Annual)",      fmt_currency(d.get("revenue")))

# ══ TAB 4 — ASSET ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Three-Slice Valuation Summary — Ch. 3</div>', unsafe_allow_html=True)
    sv1, sv2, sv3, sv4 = st.columns(4)
    sv1.metric("① Asset Value",    fmt_currency(v["reproduction_cost"]))
    sv2.metric("② EPV (Equity)",   fmt_currency(v["epv_total"]))
    sv3.metric("③ PV w/ Growth",   fmt_currency(v["pv_with_growth"]) if v["pv_with_growth"] else "N/A")
    sv4.metric("Market Cap",       fmt_currency(v["market_cap"]) if v["market_cap"] else "Enter price →")

    if v["epv_total"] and v["reproduction_cost"]:
        ratio = safe_div(v["epv_total"], v["reproduction_cost"])
        if ratio and ratio > 1:
            st.success(f"✅ EPV ({fmt_currency(v['epv_total'])}) > Reproduction Cost ({fmt_currency(v['reproduction_cost'])}) by {fmt_x(ratio)} — franchise confirmed.")
        elif ratio and ratio < 0.8:
            st.error("🔴 EPV < Reproduction Cost — poor management or industry decline (Ch. 3).")
        else:
            st.warning("⚠️ EPV ≈ Reproduction Cost — no clear franchise.")

    st.markdown('<div class="section-header">Franchise Value — Ch. 5</div>', unsafe_allow_html=True)
    fv1, fv2, fv3 = st.columns(3)
    fv1.metric("EPV (Equity)",      fmt_currency(v["epv_total"]))
    fv2.metric("Reproduction Cost", fmt_currency(v["reproduction_cost"]))
    fv3.metric("Franchise Value",   fmt_currency(v["franchise_value"]))
    if v.get("franchise_value") and v["epv_total"]:
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
    rc4.metric("ROIC",             fmt_pct(d.get("roic")))

    st.markdown('<div class="section-header">Cap Rate — Ch. 10 (Gabelli)</div>', unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("EBITDA − Capex",  fmt_currency(v["operating_cf_for_cap"]))
    ca2.metric("Enterprise Value",fmt_currency(v["ev"]) if v["ev"] else "Enter price →")
    ca3.metric("Cap Rate",        fmt_pct(v["cap_rate"]) if v["cap_rate"] else "Enter price →")
    if v["cap_rate"]:
        st.markdown("**Signal:** " + cap_signal(v["cap_rate"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sonkin Adjusted P/E — Ch. 16</div>', unsafe_allow_html=True)
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Net Cash",        fmt_currency(v["net_cash"]))
    sp2.metric("Op. Market Cap",  fmt_currency((v["market_cap"] or 0) - v["net_cash"]) if v["market_cap"] else "Enter price →")
    sp3.metric("Sonkin Adj. P/E", fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "Enter price →")

# ══ TAB 5 — SUMMARY ════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 🎯 Summary: {d['name']} [{ticker}]")
    st.caption(f"R = {fmt_pct(r)}  |  G = {fmt_pct(g)}  |  FY{d.get('fiscal_year','?')} 10-K  |  {pd.Timestamp.now().strftime('%d %b %Y')}")

    summary_rows = [
        ("Market",    "Price",                  f"${manual_price:,.2f}" if manual_price else "Enter →", "—"),
        ("Market",    "Market Cap",              fmt_currency(v["market_cap"]) if v["market_cap"] else "Enter →", "—"),
        ("Market",    "Enterprise Value",        fmt_currency(v["ev"]) if v["ev"] else "Enter →", "—"),
        ("Valuation", "① Reproduction Cost",    fmt_currency(v["reproduction_cost"]), "—"),
        ("Valuation", "② EPV / Share",           f"${v['epv_per_share']:,.2f}" if v["epv_per_share"] else "N/A", "—"),
        ("Valuation", "③ PV w/ Growth / Share",  f"${v['pv_per_share']:,.2f}"  if v["pv_per_share"]  else "N/A", "—"),
        ("Valuation", "Margin of Safety",        fmt_pct(v["mos_pct"]) if v["mos_pct"] else "Enter price →",
                                                 ("✅ BUY"  if (v["mos_pct"] or 0) >= 0.33
                                                  else ("⚠️ HOLD" if (v["mos_pct"] or 0) >= 0.10
                                                  else ("🔴 SELL" if v["mos_pct"] is not None else "—")))),
        ("Valuation", "DDM / Share",             f"${v['ddm_per_share']:,.2f}" if v["ddm_per_share"] else "N/A", "—"),
        ("Valuation", "P/E Ratio",               fmt_x(v["pe_ratio"]) if v["pe_ratio"] else "Enter price →", "—"),
        ("Growth",    "ROIC",                    fmt_pct(d.get("roic")) if d.get("roic") else "N/A",
                                                 "✅ ROC>R" if (d.get("roic") or 0) > r
                                                 else ("⚠️ ROC≈R" if abs((d.get("roic") or 0) - r) < 0.02 else "🔴 ROC<R")),
        ("Growth",    "Growth Factor F",         fmt_x(v["growth_factor_F"]) if v["growth_factor_F"] else "N/A",
                                                 "✅ Value" if (v["growth_factor_F"] or 0) > 1 else "⚠️ Neutral"),
        ("Growth",    "Revenue Growth",          fmt_pct(d.get("revenue_growth")), "—"),
        ("Growth",    "Earnings Growth",         fmt_pct(d.get("earnings_growth")), "—"),
        ("Asset",     "Franchise Value",         fmt_currency(v["franchise_value"]),
                                                 "✅ Moat" if (v["franchise_value"] or 0) > 0 else "⚠️ None"),
        ("Asset",     "Cap Rate",                fmt_pct(v["cap_rate"]) if v["cap_rate"] else "Enter price →",
                                                 ("✅ >8%" if (v["cap_rate"] or 0) >= 0.08
                                                  else ("⚠️ 5–8%" if (v["cap_rate"] or 0) >= 0.05
                                                  else ("🔴 <5%" if v["cap_rate"] is not None else "—")))),
        ("Asset",     "Sonkin Adj. P/E",         fmt_x(v["sonkin_pe"]) if v["sonkin_pe"] else "Enter price →", "—"),
    ]

    df_summary = pd.DataFrame(summary_rows, columns=["Category", "Metric", "Value", "Signal"])

    def style_signal(val):
        if "✅" in str(val): return "color: #00e676; font-weight: 600"
        if "⚠️" in str(val): return "color: #ffd740; font-weight: 600"
        if "🔴" in str(val): return "color: #ff5252; font-weight: 600"
        return "color: #888"

    st.dataframe(df_summary.style.map(style_signal, subset=["Signal"]),
                 use_container_width=True, hide_index=True, height=620)

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
        "Safety hierarchy: Asset Value → EPV → PV with Growth. "
        "Ideal buy: Price < EPV < PV with Growth. "
        "'The greater the margin of safety, the lower the risk.' — Ch. 3"
    ), unsafe_allow_html=True)
    st.caption("⚠️ Research only — not financial advice. Verify all data independently.")
