import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Try numpy_financial for IRR
try:
    import numpy_financial as npf
    HAS_NP_FIN = True
except Exception:
    HAS_NP_FIN = False

# -----------------------------
# Base model (AED)
# -----------------------------
YEARS = np.array([2025, 2026, 2027, 2028, 2029, 2030], dtype=int)

# Base operating net cashflow BEFORE financing (from your latest table)
BASE_NCF_AED = pd.Series(
    [-401_878, -814_864, -342_575, 520_062, 2_433_266, 4_940_698],
    index=YEARS, dtype=float
)

# Reference anchors
REF_ASK_AED = 1_975_023.0
PEAK_DEBT_PURE_REF_AED = 2_656_576.0      # 100 percent debt, at REF_ASK
PEAK_DEBT_HYB_REF_AED  = 1_062_630.0      # 40 percent debt in hybrid, at REF_ASK

# -----------------------------
# Page config and global CSS
# -----------------------------
st.set_page_config(page_title="Investor and Founder Financial Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1300px;}
      /* Typography */
      h1, h2, h3, h4 {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; letter-spacing: 0.1px;}
      .stTabs [data-baseweb="tab-list"] {gap: 8px;}
      .stTabs [data-baseweb="tab"] {
          background: #f7f7fb; padding: 8px 14px; border-radius: 10px;
          border: 1px solid rgba(0,0,0,0.06);
      }
      .stTabs [aria-selected="true"] {
          background: #ffffff !important; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
      }
      /* Cards */
      .card {
          border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px 16px; background:#ffffff;
          box-shadow:0 1px 5px rgba(0,0,0,0.05); height:100%;
      }
      .card-label { font-size:0.85rem; color:#666; margin-bottom:6px; white-space:normal; word-wrap:break-word; }
      .card-value { font-size:1.15rem; font-weight:700; color:#111; line-height:1.35; white-space:normal; word-wrap:break-word; }
      .card-sub   { font-size:0.8rem; color:#555; margin-top:6px; white-space:normal; word-wrap:break-word; }
      /* Footnote */
      .note {
          font-size: 0.85rem; color:#555; background:#f9fafb; border:1px solid rgba(0,0,0,0.06);
          padding:10px 12px; border-radius:10px;
      }
      /* Dataframe tweak */
      .dataframe thead tr th { font-size: 12px; }
      .dataframe tbody tr td { font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💰 Investor and Founder Financial Dashboard")

# -----------------------------
# Helpers
# -----------------------------
def fmt_int(n):
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return str(n)

def fmt_money(n, curr):
    try:
        return f"{int(round(float(n))):,} {curr}"
    except Exception:
        return f"{n} {curr}"

def info_card(label, value_str, subtext=None):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value">{value_str}</div>
          {f'<div class="card-sub">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def two_line_card(label, big_value_str, small_value_str):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value">{big_value_str}</div>
          <div class="card-sub">{small_value_str}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def convert_amount(aed_value, currency, fx_aed_per_usd):
    return aed_value if currency == "AED" else aed_value / fx_aed_per_usd

def convert_series(series_aed, currency, fx_aed_per_usd):
    return series_aed.apply(lambda v: convert_amount(v, currency, fx_aed_per_usd))

def npv(rate, cashflows):
    t = np.arange(len(cashflows))
    return float(np.sum(cashflows / ((1 + rate) ** t)))

def irr(cashflows):
    if HAS_NP_FIN:
        try:
            return float(npf.irr(cashflows))
        except Exception:
            pass
    low, high = -0.999, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        val = np.sum(cashflows / ((1 + mid) ** np.arange(len(cashflows))))
        if val > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def irr_for_exit(investment, years_from_now, exit_value):
    if years_from_now <= 0:
        return np.nan
    cfs = np.zeros(years_from_now + 1)
    cfs[0] = -investment
    cfs[-1] = exit_value
    return irr(cfs)

def df_with_commas(df):
    """Return a copy of df with all numeric values rendered as strings with commas."""
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].apply(lambda x: fmt_int(x) if isinstance(x, (int, float, np.number)) else x)
    return df2

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.subheader("Currency")
    currency = st.selectbox("Display currency", ["AED", "USD"], index=0)
    fx = st.number_input("AED per 1 USD", min_value=0.0001, value=3.67, step=0.01, format="%.2f")

    st.markdown("---")
    st.subheader("Round Inputs (AED base)")
    current_val_aed = st.number_input("Current valuation (AED, pre money)", min_value=10_000.0,
                                      value=2_478_629.0, step=50_000.0, format="%.2f")
    ask_aed = st.number_input("Investment amount (AED)", min_value=10_000.0,
                              value=REF_ASK_AED, step=50_000.0, format="%.2f")
    discount_rate = st.number_input("Discount rate for NPV (decimal)", min_value=0.0, max_value=1.0,
                                    value=0.12, step=0.01, format="%.2f")

    st.markdown("---")
    st.subheader("Debt Assumptions")
    cash_interest_rate = st.slider("Cash interest rate for 2028 to 2030", min_value=0, max_value=40, value=14, step=1) / 100.0
    bullet_year_default = 2030
    bullet_year = st.selectbox("Bullet repayment year", options=list(YEARS), index=list(YEARS).index(bullet_year_default))

    st.markdown("---")
    st.subheader("Equity Exit Modelling")
    exit_mode = st.radio("Exit valuation method", ["Valuation CAGR", "ARR × EV/ARR Multiple"], index=0)
    if exit_mode == "Valuation CAGR":
        cagr = st.slider("Valuation CAGR percent", min_value=0, max_value=150, value=40, step=5) / 100.0
        arr_now_aed = arr_growth = ev_arr_mult = None
    else:
        arr_input_display = st.number_input(f"Current ARR ({currency})", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.2f")
        arr_now_aed = arr_input_display if currency == "AED" else arr_input_display * fx
        arr_growth = st.slider("ARR growth percent per year", min_value=0, max_value=200, value=60, step=5) / 100.0
        ev_arr_mult = st.slider("EV/ARR multiple", min_value=1, max_value=25, value=8, step=1)
        cagr = None

    st.markdown("---")
    st.subheader("Hybrid Split")
    hybrid_equity_pct = st.slider("Hybrid equity percent", min_value=0, max_value=100, value=60, step=5) / 100.0

    st.markdown("---")
    st.subheader("Founder Assumptions")
    start_cash_display = st.number_input(f"Starting cash ({currency})", min_value=0.0, value=0.0, step=50_000.0, format="%.2f")
    buffer_display     = st.number_input(f"Minimum cash buffer ({currency})", min_value=0.0, value=0.0, step=25_000.0, format="%.2f")

# Internal conversions for founder inputs
start_cash_aed = start_cash_display if currency == "AED" else start_cash_display * fx
buffer_aed     = buffer_display     if currency == "AED" else buffer_display * fx

# Stake rule shown as investment / pre money (can optionally show post money stake if you want a toggle)
equity_pct_round = min(1.0, ask_aed / current_val_aed)
post_money_aed = current_val_aed + ask_aed

# -----------------------------
# Scenario engines (AED)
# -----------------------------
def company_net_cf_equity_aed(ask_aed: float) -> pd.Series:
    eq_in = pd.Series(0.0, index=YEARS)
    eq_in.iloc[0] = ask_aed
    return BASE_NCF_AED + eq_in

def company_net_cf_debt_aed(ask_aed: float, interest_rate: float, bullet_year: int) -> tuple[pd.Series, float]:
    ratio = ask_aed / REF_ASK_AED
    peak_debt = PEAK_DEBT_PURE_REF_AED * ratio

    debt_in = pd.Series(0.0, index=YEARS); debt_in.iloc[0] = ask_aed
    cash_int = pd.Series(0.0, index=YEARS)
    cash_int.loc[YEARS[YEARS >= 2028]] = -interest_rate * peak_debt

    bullet = pd.Series(0.0, index=YEARS)
    if bullet_year in bullet.index:
        bullet.loc[bullet_year] = -peak_debt

    company_cf = BASE_NCF_AED + debt_in + cash_int + bullet
    return company_cf, peak_debt

def company_net_cf_hybrid_aed(ask_aed: float, equity_pct: float, interest_rate: float, bullet_year: int) -> tuple[pd.Series, float]:
    equity_share = equity_pct
    debt_share = 1.0 - equity_share

    eq_in = pd.Series(0.0, index=YEARS);   eq_in.iloc[0]   = ask_aed * equity_share
    debt_in = pd.Series(0.0, index=YEARS); debt_in.iloc[0] = ask_aed * debt_share

    # scale hybrid peak debt from 40 percent reference anchor safely
    if debt_share == 0:
        peak_debt_h = 0.0
    else:
        scale = (ask_aed / REF_ASK_AED) * (debt_share / 0.40)
        peak_debt_h = PEAK_DEBT_HYB_REF_AED * scale

    cash_int = pd.Series(0.0, index=YEARS)
    cash_int.loc[YEARS[YEARS >= 2028]] = -interest_rate * peak_debt_h

    bullet = pd.Series(0.0, index=YEARS)
    if bullet_year in bullet.index:
        bullet.loc[bullet_year] = -peak_debt_h

    company_cf = BASE_NCF_AED + eq_in + debt_in + cash_int + bullet
    return company_cf, peak_debt_h

# Investor CFs for Debt/Hybrid (debt leg)
def investor_cf_debt_aed(ask_aed: float, interest_rate: float, bullet_year: int) -> pd.Series:
    ratio = ask_aed / REF_ASK_AED
    peak_debt = PEAK_DEBT_PURE_REF_AED * ratio
    inv = pd.Series(0.0, index=YEARS); inv.iloc[0] = -ask_aed
    cash_int = pd.Series(0.0, index=YEARS); cash_int.loc[YEARS[YEARS >= 2028]] = +interest_rate * peak_debt
    bullet = pd.Series(0.0, index=YEARS)
    if bullet_year in bullet.index: bullet.loc[bullet_year] = +peak_debt
    return inv + cash_int + bullet

def investor_cf_hybrid_debt_leg_aed(ask_aed: float, equity_pct: float, interest_rate: float, bullet_year: int) -> pd.Series:
    _, peak_debt_h = company_net_cf_hybrid_aed(ask_aed, equity_pct, interest_rate, bullet_year)
    inv = pd.Series(0.0, index=YEARS); inv.iloc[0] = -(ask_aed * (1.0 - equity_pct))
    cash_int = pd.Series(0.0, index=YEARS); cash_int.loc[YEARS[YEARS >= 2028]] = +interest_rate * peak_debt_h
    bullet = pd.Series(0.0, index=YEARS)
    if bullet_year in bullet.index: bullet.loc[bullet_year] = +peak_debt_h
    return inv + cash_int + bullet

# -----------------------------
# Equity Exit Table (AED math)
# -----------------------------
def build_equity_exit_table_aed(current_valuation_aed: float,
                                investment_aed: float,
                                equity_pct: float,
                                mode: str,
                                cagr: float | None,
                                arr_now_aed: float | None,
                                arr_growth: float | None,
                                ev_arr_mult: float | None) -> pd.DataFrame:
    rows = []
    for i, y in enumerate(YEARS):
        if mode == "Valuation CAGR":
            val_aed = current_valuation_aed * ((1 + (cagr or 0.0)) ** i)
        else:
            arr_y = (arr_now_aed or 0.0) * ((1 + (arr_growth or 0.0)) ** i)
            val_aed = (ev_arr_mult or 0.0) * arr_y

        equity_value_aed = equity_pct * val_aed
        moic = equity_value_aed / investment_aed if investment_aed > 0 else np.nan
        irr_v = np.nan if i == 0 else irr_for_exit(investment_aed, i, equity_value_aed) * 100.0

        rows.append([y, val_aed, equity_value_aed, equity_pct, moic, irr_v])

    return pd.DataFrame(rows, columns=[
        "Year", "Company Valuation (AED)", "Equity Value (AED)", "Equity %", "MOIC (x)", "IRR (%)"
    ]).set_index("Year")

# -----------------------------
# Build equity exit table then currency safe view
# -----------------------------
equity_exit_aed = build_equity_exit_table_aed(
    current_valuation_aed=current_val_aed,
    investment_aed=ask_aed,
    equity_pct=equity_pct_round,
    mode=exit_mode,
    cagr=cagr,
    arr_now_aed=arr_now_aed, arr_growth=arr_growth, ev_arr_mult=ev_arr_mult
)

if currency == "USD":
    equity_exit_df = equity_exit_aed.copy()
    equity_exit_df["Company Valuation (USD)"] = convert_series(equity_exit_aed["Company Valuation (AED)"], currency, fx)
    equity_exit_df["Equity Value (USD)"] = convert_series(equity_exit_aed["Equity Value (AED)"], currency, fx)
else:
    equity_exit_df = equity_exit_aed.copy()

default_exit_year = int(equity_exit_df.index[-1])
if currency == "USD":
    sel_equity_value_default = float(equity_exit_df.loc[default_exit_year, "Equity Value (USD)"])
else:
    sel_equity_value_default = float(equity_exit_df.loc[default_exit_year, "Equity Value (AED)"])

CUR = currency
ask_disp = convert_amount(ask_aed, CUR, fx)
current_val_disp = convert_amount(current_val_aed, CUR, fx)
post_money_disp = convert_amount(post_money_aed, CUR, fx)

# -----------------------------
# Build company cash paths (AED) for tabs and charts
# -----------------------------
cf_equity_aed = company_net_cf_equity_aed(ask_aed)
cf_debt_aed, peak_debt_aed = company_net_cf_debt_aed(ask_aed, cash_interest_rate, bullet_year)
cf_hyb_aed, peak_debt_hyb_aed = company_net_cf_hybrid_aed(ask_aed, hybrid_equity_pct, cash_interest_rate, bullet_year)

end_equity_aed = cf_equity_aed.cumsum() + start_cash_aed
end_debt_aed   = cf_debt_aed.cumsum()   + start_cash_aed
end_hyb_aed    = cf_hyb_aed.cumsum()    + start_cash_aed

# Display currency versions
cf_equity = convert_series(cf_equity_aed, CUR, fx)
cf_debt   = convert_series(cf_debt_aed,   CUR, fx)
cf_hyb    = convert_series(cf_hyb_aed,    CUR, fx)
end_equity = convert_series(end_equity_aed, CUR, fx)
end_debt   = convert_series(end_debt_aed,   CUR, fx)
end_hyb    = convert_series(end_hyb_aed,    CUR, fx)

# -----------------------------
# Tabs
# -----------------------------
tab_eq, tab_debt, tab_hyb, tab_founder, tab_compare = st.tabs(
    ["100 percent Equity (Exit based)", "100 percent Debt", "Hybrid", "Founder", "Compare"]
)

# ---------- Equity tab ----------
with tab_eq:
    st.subheader("100 percent Equity (Exit based)")

    c1, c2, c3 = st.columns(3)
    with c1:
        two_line_card("How much you invest today", fmt_money(ask_disp, CUR),
                      f"Stake equals {equity_pct_round*100:.2f}% of company")
    with c2:
        two_line_card(f"How much you could get back if exit in {default_exit_year}",
                      fmt_money(sel_equity_value_default, CUR),
                      f"MOIC {equity_exit_aed.loc[default_exit_year, 'MOIC (x)']:.2f}x • "
                      f"IRR {equity_exit_aed.loc[default_exit_year, 'IRR (%)']:.2f}%")
    with c3:
        info_card("When", str(default_exit_year), "Default equals last modeled year")

    k1, k2, k3 = st.columns(3)
    with k1: two_line_card("Current valuation", fmt_money(current_val_disp, CUR), "Pre money")
    with k2: two_line_card("Investment", fmt_money(ask_disp, CUR), "Raise amount")
    with k3: two_line_card("Post money", fmt_money(post_money_disp, CUR), "Pre plus investment")

    st.markdown("#### Exit valuation path")
    chosen_exit_year = st.select_slider("Choose exit year", options=list(equity_exit_df.index), value=default_exit_year)
    if CUR == "USD":
        chosen_equity_value = float(equity_exit_df.loc[chosen_exit_year, "Equity Value (USD)"])
    else:
        chosen_equity_value = float(equity_exit_df.loc[chosen_exit_year, "Equity Value (AED)"])
    chosen_moic = float(equity_exit_aed.loc[chosen_exit_year, "MOIC (x)"])
    chosen_irr  = float(equity_exit_aed.loc[chosen_exit_year, "IRR (%)"])
    st.caption(f"Selected exit year equals {chosen_exit_year}. Equity value equals {fmt_money(chosen_equity_value, CUR)}. MOIC {chosen_moic:.2f}x, IRR {chosen_irr:.2f}%.")

    # Table formatted
    df_show = equity_exit_df.copy()
    money_cols = ["Company Valuation (AED)", "Equity Value (AED)"]
    if CUR == "USD":
        money_cols = ["Company Valuation (USD)", "Equity Value (USD)"]

    # Human readable
    for col in money_cols:
        df_show[col] = df_show[col].map(lambda x: fmt_int(x))
    df_show["Equity %"] = equity_exit_aed["Equity %"].map(lambda x: f"{x*100:.2f}%")
    df_show["MOIC (x)"] = equity_exit_aed["MOIC (x)"].map(lambda x: f"{x:.2f}")
    df_show["IRR (%)"] = equity_exit_aed["IRR (%)"].map(lambda x: "n.a." if pd.isna(x) else f"{x:.2f}%")
    st.dataframe(df_show)

    # Chart
    fig, ax = plt.subplots()
    y_val_col = "Company Valuation (USD)" if CUR == "USD" else "Company Valuation (AED)"
    y_eq_col  = "Equity Value (USD)" if CUR == "USD" else "Equity Value (AED)"
    ax.plot(equity_exit_df.index, equity_exit_df[y_val_col], marker="o", label=f"Company valuation ({CUR})")
    ax.plot(equity_exit_df.index, equity_exit_df[y_eq_col], marker="o", label=f"Your equity value ({CUR})")
    ax.set_title(f"Valuation and Equity value by exit year ({CUR})")
    ax.set_xlabel("Year"); ax.set_ylabel(CUR)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)

# ---------- Debt tab ----------
with tab_debt:
    st.subheader("100 percent Debt")

    inv_cf_debt = investor_cf_debt_aed(ask_aed, cash_interest_rate, bullet_year)
    inv_cf_debt_disp = convert_series(inv_cf_debt, CUR, fx)

    invested = -min(inv_cf_debt_disp.iloc[0], 0.0)
    returned = inv_cf_debt_disp[inv_cf_debt_disp > 0].sum()

    d1, d2, d3 = st.columns(3)
    with d1:
        two_line_card("How much you invest today", fmt_money(invested, CUR), "Loan principal at t0")
    with d2:
        two_line_card("How much you will get back", fmt_money(returned, CUR), "Interest plus principal")
    with d3:
        info_card("When", f"Interest 2028 to {YEARS[-1]}", f"Bullet in {bullet_year}")

    st.markdown("#### Company cashflows")
    st.dataframe(df_with_commas(cf_debt.to_frame(f"Company net cashflow ({CUR})")))

    fig, ax = plt.subplots()
    ax.bar(cf_debt.index.astype(str), cf_debt.values)
    ax.set_title(f"Company net cashflows ({CUR})"); ax.set_xlabel("Year"); ax.set_ylabel(CUR)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)

# ---------- Hybrid tab ----------
with tab_hyb:
    st.subheader("Hybrid")

    inv_cf_hyb_debt = investor_cf_hybrid_debt_leg_aed(ask_aed, hybrid_equity_pct, cash_interest_rate, bullet_year)
    inv_cf_hyb_debt_disp = convert_series(inv_cf_hyb_debt, CUR, fx)

    invested_h = -min(inv_cf_hyb_debt_disp.iloc[0], 0.0)
    returned_h = inv_cf_hyb_debt_disp[inv_cf_hyb_debt_disp > 0].sum()

    h1, h2, h3 = st.columns(3)
    with h1:
        two_line_card("How much you invest today (Debt leg)", fmt_money(invested_h, CUR), "Loan principal at t0")
    with h2:
        two_line_card("How much you will get back (Debt leg)", fmt_money(returned_h, CUR), "Interest plus principal")
    with h3:
        info_card("When", f"Interest 2028 to {YEARS[-1]}", f"Bullet in {bullet_year}")

    st.markdown("#### Company cashflows")
    st.dataframe(df_with_commas(cf_hyb.to_frame(f"Company net cashflow ({CUR})")))

    fig, ax = plt.subplots()
    ax.bar(cf_hyb.index.astype(str), cf_hyb.values)
    ax.set_title(f"Company net cashflows ({CUR})"); ax.set_xlabel("Year"); ax.set_ylabel(CUR)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)

# ---------- Founder tab ----------
with tab_founder:
    st.subheader("Founder view: cash, bullet coverage, and timing of next raise")

    scen_choice = st.selectbox("Scenario to evaluate", ["Equity", "Debt", "Hybrid"], index=1)

    if scen_choice == "Equity":
        cf_use_aed = company_net_cf_equity_aed(ask_aed)
        end_use_aed = cf_use_aed.cumsum() + start_cash_aed
        bullet_year_use = None
        bullet_amt_aed = 0.0
    elif scen_choice == "Debt":
        cf_use_aed, bullet_amt_aed = company_net_cf_debt_aed(ask_aed, cash_interest_rate, bullet_year)
        end_use_aed = cf_use_aed.cumsum() + start_cash_aed
        bullet_year_use = int(bullet_year)
    else:
        cf_use_aed, bullet_amt_aed = company_net_cf_hybrid_aed(ask_aed, hybrid_equity_pct, cash_interest_rate, bullet_year)
        end_use_aed = cf_use_aed.cumsum() + start_cash_aed
        bullet_year_use = int(bullet_year)

    cf_use = convert_series(cf_use_aed, CUR, fx)
    end_use = convert_series(end_use_aed, CUR, fx)
    bullet_amt = convert_amount(bullet_amt_aed, CUR, fx)

    c1, c2, c3 = st.columns(3)
    with c1:
        two_line_card("Starting cash", fmt_money(start_cash_display, CUR),
                      f"Min buffer equals {fmt_money(buffer_display, CUR)}")
    with c2:
        two_line_card(f"Cash at end of {end_use.index[0]}", fmt_money(end_use.iloc[0], CUR),
                      f"Year equals {end_use.index[0]}")
    with c3:
        two_line_card(f"Cash at end of {end_use.index[-1]}", fmt_money(end_use.iloc[-1], CUR),
                      f"Year equals {end_use.index[-1]}")

    # Bullet coverage snapshot
    if bullet_year_use is None:
        info_card("Bullet coverage", "Not applicable")
    else:
        cash_by_bullet = float(end_use.loc[bullet_year_use])
        status = "Yes" if cash_by_bullet >= bullet_amt else "No"
        two_line_card("Bullet coverage", status,
                      f"Cash in {bullet_year_use}: {fmt_money(cash_by_bullet, CUR)} • Bullet: {fmt_money(bullet_amt, CUR)}")

    st.markdown("#### Ending cash by year")
    df_founder = pd.DataFrame({
        "Year": end_use.index,
        f"Company net CF ({CUR})": cf_use.values,
        f"Ending cash ({CUR})": end_use.values
    }).set_index("Year")
    st.dataframe(df_with_commas(df_founder))

    fig1, ax1 = plt.subplots()
    ax1.plot(end_use.index, end_use.values, marker="o")
    ax1.axhline(buffer_display, linestyle="--", linewidth=1)
    ax1.set_title(f"Ending cash versus buffer ({CUR})")
    ax1.set_xlabel("Year"); ax1.set_ylabel(CUR)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax1.grid(True, alpha=0.25)
    st.pyplot(fig1)

    # Future raise recommendation
    st.markdown("#### Future raise recommendation for bullet coverage and positive cashflow")

    # 1) First year where ending cash dips below buffer
    shortfall_year = None
    for yr in end_use_aed.index:
        if end_use_aed.loc[yr] < buffer_aed:
            shortfall_year = int(yr)
            break

    if shortfall_year is None:
        st.success("You are self sustained through 2030 under current funding and assumptions.")
    else:
        # 2) Estimate next 12 month need using the following year's base NCF magnitude as proxy
        next_year = shortfall_year + 1
        if next_year in BASE_NCF_AED.index:
            next_year_expense = abs(BASE_NCF_AED.loc[next_year])
        else:
            next_year_expense = abs(BASE_NCF_AED.loc[shortfall_year])

        # 3) Bullet alignment
        bullet_due = (bullet_year_use is not None) and (bullet_year_use in [shortfall_year, shortfall_year + 1])
        bullet_add_aed = bullet_amt_aed if bullet_due else 0.0

        # 4) Amount to raise
        needed_raise_aed = max(0.0, (buffer_aed - end_use_aed.loc[shortfall_year]) + next_year_expense + bullet_add_aed)
        needed_raise_disp = convert_amount(needed_raise_aed, CUR, fx)

        when_text = f"{shortfall_year} (before cash dips below buffer)"
        if bullet_add_aed > 0:
            when_text += f" - aligns with {bullet_year_use} bullet repayment"

        st.warning(f"Expected cash shortfall in {shortfall_year}.")
        two_line_card("Ideal time for next raise", when_text, "Target to restore 12 month runway and cover obligations")
        two_line_card("Recommended raise amount", fmt_money(needed_raise_disp, CUR),
                      f"Includes bullet coverage equals {fmt_money(convert_amount(bullet_add_aed, CUR, fx), CUR)}")

    # Optional quick NPVs for the company path under this scenario
    st.markdown("#### Quick NPV of company net cashflows (informational)")
    try:
        npv_value = npv(discount_rate, cf_use_aed.values)
        info_card("Company NPV at selected discount rate", fmt_money(convert_amount(npv_value, CUR, fx), CUR),
                  f"Discount rate equals {discount_rate:.2%}")
    except Exception:
        info_card("Company NPV", "Not available")

# ---------- Compare tab ----------
with tab_compare:
    st.subheader("Quick compare of company net cashflows")
    fig2, ax2 = plt.subplots()
    ax2.plot(cf_equity.index, cf_equity.values, marker="o", label="Equity")
    ax2.plot(cf_debt.index,   cf_debt.values,   marker="o", label="Debt")
    ax2.plot(cf_hyb.index,    cf_hyb.values,    marker="o", label="Hybrid")
    ax2.set_title(f"Annual company net cashflows ({CUR})")
    ax2.set_xlabel("Year"); ax2.set_ylabel(CUR); ax2.legend()
    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax2.grid(True, alpha=0.25)
    st.pyplot(fig2)

# -----------------------------
# Footnotes and assumptions
# -----------------------------
st.markdown(
    """
    <div class="note">
      <b>Notes and assumptions</b><br>
      Base operating cashflows are pre financing. Debt interest is applied from 2028 onward to the selected bullet year inclusive.
      Bullet repayment year is user selectable. Hybrid peak debt scales from a 40 percent debt reference anchor.
      Equity stake is displayed as investment divided by pre money for clarity. IRR is not defined for year 0.
      All AED to USD conversions use the sidebar exchange rate.
    </div>
    """,
    unsafe_allow_html=True
)
