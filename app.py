import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Base operating net cashflow BEFORE financing (from your table)
BASE_NCF_AED = pd.Series(
    [-401_878, -814_864, -342_575, 520_062, 2_433_266, 4_940_698],
    index=YEARS, dtype=float
)

# Debt mechanics (per your scenarios)
REF_ASK_AED = 1_975_023.0
PEAK_DEBT_PURE_REF_AED = 2_656_576.0      # 100% debt, at REF_ASK
PEAK_DEBT_HYB_REF_AED  = 1_062_630.0      # 40% debt in hybrid, at REF_ASK
CASH_INTEREST_RATE = 0.14                 # cash interest assumed 2028–2030

# -----------------------------
# Formatting & helpers
# -----------------------------
def fmt_int(n):
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return str(n)

def fmt_money(n, curr):
    return f"{fmt_int(n)} {curr}"

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

# Finance helpers
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
    cfs = np.zeros(years_from_now + 1)
    cfs[0] = -investment
    cfs[-1] = exit_value
    return irr(cfs)

# -----------------------------
# Scenario engines (AED)
# -----------------------------
def company_net_cf_equity_aed(ask_aed: float) -> pd.Series:
    """Company net CF for 100% equity: base + equity inflow at 2025."""
    eq_in = pd.Series(0.0, index=YEARS)
    eq_in.iloc[0] = ask_aed
    return BASE_NCF_AED + eq_in

def company_net_cf_debt_aed(ask_aed: float) -> tuple[pd.Series, float]:
    """Company net CF for 100% debt, plus peak debt (bullet)."""
    ratio = ask_aed / REF_ASK_AED
    peak_debt = PEAK_DEBT_PURE_REF_AED * ratio

    debt_in = pd.Series(0.0, index=YEARS); debt_in.iloc[0] = ask_aed
    cash_int = pd.Series(0.0, index=YEARS)
    cash_int.loc[YEARS[YEARS >= 2028]] = -CASH_INTEREST_RATE * peak_debt
    bullet = pd.Series(0.0, index=YEARS); bullet.iloc[-1] = -peak_debt

    company_cf = BASE_NCF_AED + debt_in + cash_int + bullet
    return company_cf, peak_debt

def company_net_cf_hybrid_aed(ask_aed: float, equity_pct: float) -> tuple[pd.Series, float]:
    """Hybrid company net CF (equity+debt inflow, cash interest from 2028, bullet in 2030)."""
    equity_share = equity_pct
    debt_share = 1.0 - equity_share

    eq_in = pd.Series(0.0, index=YEARS);   eq_in.iloc[0]   = ask_aed * equity_share
    debt_in = pd.Series(0.0, index=YEARS); debt_in.iloc[0] = ask_aed * debt_share

    # scale hybrid peak debt from 40% reference
    scale = (ask_aed / REF_ASK_AED) * (debt_share / 0.40 if 0.40 > 0 else 0.0)
    peak_debt_h = PEAK_DEBT_HYB_REF_AED * scale

    cash_int = pd.Series(0.0, index=YEARS)
    cash_int.loc[YEARS[YEARS >= 2028]] = -CASH_INTEREST_RATE * peak_debt_h
    bullet = pd.Series(0.0, index=YEARS); bullet.iloc[-1] = -peak_debt_h

    company_cf = BASE_NCF_AED + eq_in + debt_in + cash_int + bullet
    return company_cf, peak_debt_h

# Investor CFs for Debt/Hybrid tabs (debt leg)
def investor_cf_debt_aed(ask_aed: float) -> pd.Series:
    ratio = ask_aed / REF_ASK_AED
    peak_debt = PEAK_DEBT_PURE_REF_AED * ratio
    inv = pd.Series(0.0, index=YEARS); inv.iloc[0] = -ask_aed
    cash_int = pd.Series(0.0, index=YEARS); cash_int.loc[YEARS[YEARS >= 2028]] = +CASH_INTEREST_RATE * peak_debt
    bullet = pd.Series(0.0, index=YEARS); bullet.iloc[-1] = +peak_debt
    return inv + cash_int + bullet

def investor_cf_hybrid_debt_leg_aed(ask_aed: float, equity_pct: float) -> pd.Series:
    _, peak_debt_h = company_net_cf_hybrid_aed(ask_aed, equity_pct)
    inv = pd.Series(0.0, index=YEARS); inv.iloc[0] = -(ask_aed * (1.0 - equity_pct))
    cash_int = pd.Series(0.0, index=YEARS); cash_int.loc[YEARS[YEARS >= 2028]] = +CASH_INTEREST_RATE * peak_debt_h
    bullet = pd.Series(0.0, index=YEARS); bullet.iloc[-1] = +peak_debt_h
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
        irr_v = irr_for_exit(investment_aed, i, equity_value_aed) * 100.0

        rows.append([y, val_aed, equity_value_aed, equity_pct, moic, irr_v])

    return pd.DataFrame(rows, columns=[
        "Year", "Company Valuation (AED)", "Equity Value (AED)", "Equity %", "MOIC (x)", "IRR (%)"
    ]).set_index("Year")

# -----------------------------
# UI shell and CSS
# -----------------------------
st.set_page_config(page_title="Funding Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .card { border:1px solid rgba(0,0,0,0.08); border-radius:12px; padding:14px 16px; background:#fff;
            box-shadow:0 1px 3px rgba(0,0,0,0.05); height:100%; }
    .card-label { font-size:0.85rem; color:#666; margin-bottom:6px; white-space:normal; word-wrap:break-word; }
    .card-value { font-size:1.1rem; font-weight:700; color:#111; line-height:1.3; white-space:normal; word-wrap:break-word; }
    .card-sub   { font-size:0.8rem; color:#555; margin-top:6px; white-space:normal; word-wrap:break-word; }
    </style>
    """, unsafe_allow_html=True
)

st.title("💰 Investor & Founder Financial Dashboard")

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.subheader("Currency")
    currency = st.selectbox("Display currency", ["AED", "USD"], index=0)
    fx = st.number_input("AED per 1 USD", min_value=0.0001, value=3.67, step=0.01, format="%.2f")

    st.markdown("---")
    st.subheader("Round Inputs (AED base)")
    current_val_aed = st.number_input("Current valuation (AED, pre-money)", min_value=10_000.0,
                                      value=2_478_629.0, step=50_000.0, format="%.2f")
    ask_aed = st.number_input("Investment amount (AED)", min_value=10_000.0,
                              value=REF_ASK_AED, step=50_000.0, format="%.2f")
    discount_rate = st.number_input("Discount rate for NPV (decimal)", min_value=0.0, max_value=1.0,
                                    value=0.12, step=0.01, format="%.2f")

    st.markdown("---")
    st.subheader("Equity Exit Modelling")
    exit_mode = st.radio("Exit valuation method", ["Valuation CAGR", "ARR × EV/ARR Multiple"], index=0)
    if exit_mode == "Valuation CAGR":
        cagr = st.slider("Valuation CAGR percent", min_value=0, max_value=150, value=40, step=5) / 100.0
        arr_now_aed = arr_growth = ev_arr_mult = None
    else:
        # ARR entered in display currency for UX; convert internally to AED
        arr_input_display = st.number_input(f"Current ARR ({currency})", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.2f")
        arr_now_aed = arr_input_display if currency == "AED" else arr_input_display * fx
        arr_growth = st.slider("ARR growth percent per year", min_value=0, max_value=200, value=60, step=5) / 100.0
        ev_arr_mult = st.slider("EV/ARR multiple", min_value=1, max_value=25, value=8, step=1)
        cagr = None

    st.markdown("---")
    st.subheader("Hybrid split")
    hybrid_equity_pct = st.slider("Hybrid equity percent", min_value=0, max_value=100, value=60, step=5) / 100.0

    st.markdown("---")
    st.subheader("Founder assumptions")
    start_cash_display = st.number_input(f"Starting cash ({currency})", min_value=0.0, value=0.0, step=50_000.0, format="%.2f")
    buffer_display     = st.number_input(f"Minimum cash buffer ({currency})", min_value=0.0, value=0.0, step=25_000.0, format="%.2f")

# Convert founder inputs for internal math
start_cash_aed = start_cash_display if currency == "AED" else start_cash_display * fx
buffer_aed     = buffer_display     if currency == "AED" else buffer_display * fx

# Stake per your rule: investment ÷ current valuation (pre-money)
equity_pct_round = min(1.0, ask_aed / current_val_aed)
post_money_aed = current_val_aed + ask_aed

# -----------------------------
# Equity exit table (AED) then currency-safe view
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

# Selected exit year default = last year
default_exit_year = int(equity_exit_df.index[-1])
if currency == "USD":
    sel_equity_value_default = float(equity_exit_df.loc[default_exit_year, "Equity Value (USD)"])
else:
    sel_equity_value_default = float(equity_exit_df.loc[default_exit_year, "Equity Value (AED)"])

# Display values
CUR = currency
ask_disp = convert_amount(ask_aed, CUR, fx)
current_val_disp = convert_amount(current_val_aed, CUR, fx)
post_money_disp = convert_amount(post_money_aed, CUR, fx)

# -----------------------------
# Build company cash paths (AED) for Founder + charts
# -----------------------------
cf_equity_aed = company_net_cf_equity_aed(ask_aed)
cf_debt_aed, peak_debt_aed = company_net_cf_debt_aed(ask_aed)
cf_hyb_aed, peak_debt_hyb_aed = company_net_cf_hybrid_aed(ask_aed, hybrid_equity_pct)

end_equity_aed = cf_equity_aed.cumsum() + start_cash_aed
end_debt_aed   = cf_debt_aed.cumsum()   + start_cash_aed
end_hyb_aed    = cf_hyb_aed.cumsum()    + start_cash_aed

# Display-currency versions
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
    ["100 percent Equity (Exit-based)", "100 percent Debt", "Hybrid", "Founder", "Compare"]
)

# ---------- Equity tab ----------
with tab_eq:
    st.subheader("100 percent Equity — Exit-based")

    # Investor trio
    c1, c2, c3 = st.columns([1,1,1])
    two_line_card("How much you invest today", fmt_money(ask_disp, CUR),
                  f"Stake = {equity_pct_round*100:.2f}% of company")
    two_line_card(f"How much you will get back (if exit in {default_exit_year})",
                  fmt_money(sel_equity_value_default, CUR),
                  f"MOIC {equity_exit_aed.loc[default_exit_year, 'MOIC (x)']:.2f}x • "
                  f"IRR {equity_exit_aed.loc[default_exit_year, 'IRR (%)']:.2f}%")
    info_card("When", str(default_exit_year))

    # Round math
    k1, k2, k3 = st.columns(3)
    with k1: two_line_card("Current Valuation", fmt_money(current_val_disp, CUR), "Pre-money")
    with k2: two_line_card("Investment", fmt_money(ask_disp, CUR), "= Raise amount")
    with k3: two_line_card("Post-money", fmt_money(post_money_disp, CUR), "Pre + Investment")

    st.markdown("#### Exit Valuation Path")
    chosen_exit_year = st.select_slider("Choose exit year", options=list(equity_exit_df.index), value=default_exit_year)
    if CUR == "USD":
        chosen_equity_value = float(equity_exit_df.loc[chosen_exit_year, "Equity Value (USD)"])
    else:
        chosen_equity_value = float(equity_exit_df.loc[chosen_exit_year, "Equity Value (AED)"])
    chosen_moic = float(equity_exit_aed.loc[chosen_exit_year, "MOIC (x)"])
    chosen_irr  = float(equity_exit_aed.loc[chosen_exit_year, "IRR (%)"])
    st.caption(f"Selected exit – Equity value: {fmt_money(chosen_equity_value, CUR)}, "
               f"MOIC {chosen_moic:.2f}x, IRR {chosen_irr:.2f}%.")

    # Table formatted
    df_show = equity_exit_df.copy()
    money_cols = ["Company Valuation (AED)", "Equity Value (AED)"]
    if CUR == "USD":
        money_cols = ["Company Valuation (USD)", "Equity Value (USD)"]
    for col in money_cols:
        df_show[col] = df_show[col].map(lambda x: fmt_int(x))
    df_show["Equity %"] = equity_exit_aed["Equity %"].map(lambda x: f"{x*100:.2f}%")
    df_show["MOIC (x)"] = equity_exit_aed["MOIC (x)"].map(lambda x: f"{x:.2f}")
    df_show["IRR (%)"] = equity_exit_aed["IRR (%)"].map(lambda x: f"{x:.2f}%")
    st.dataframe(df_show)

    # Chart
    fig, ax = plt.subplots()
    ax.plot(
        equity_exit_df.index,
        equity_exit_df["Company Valuation (USD)"] if CUR == "USD" else equity_exit_df["Company Valuation (AED)"],
        marker="o", label=f"Company Valuation ({CUR})"
    )
    ax.plot(
        equity_exit_df.index,
        equity_exit_df["Equity Value (USD)"] if CUR == "USD" else equity_exit_df["Equity Value (AED)"],
        marker="o", label=f"Your Equity Value ({CUR})"
    )
    ax.set_title(f"Valuation & Your Equity Value by Exit Year ({CUR})")
    ax.set_xlabel("Year"); ax.set_ylabel(CUR); ax.legend()
    st.pyplot(fig)

# ---------- Debt tab ----------
with tab_debt:
    st.subheader("100 percent Debt")

    inv_cf_debt = investor_cf_debt_aed(ask_aed)
    inv_cf_debt_disp = convert_series(inv_cf_debt, CUR, fx)

    # Investor trio
    invested = -min(inv_cf_debt_disp.iloc[0], 0.0)
    returned = inv_cf_debt_disp[inv_cf_debt_disp > 0].sum()
    two_line_card("How much you invest today", fmt_money(invested, CUR), "Loan principal at t0")
    two_line_card("How much you will get back", fmt_money(returned, CUR), "Interest + principal")
    info_card("When", "Interest 2028–2030 • Bullet 2030")

    st.markdown("#### Company cashflows")
    st.dataframe(cf_debt.to_frame(f"Company Net CF ({CUR})").style.format("{:,.0f}"))

    fig, ax = plt.subplots()
    ax.bar(cf_debt.index.astype(str), cf_debt.values)
    ax.set_title(f"Company Net Cashflows ({CUR})"); ax.set_xlabel("Year"); ax.set_ylabel(CUR)
    st.pyplot(fig)

# ---------- Hybrid tab ----------
with tab_hyb:
    st.subheader("Hybrid")

    inv_cf_hyb_debt = investor_cf_hybrid_debt_leg_aed(ask_aed, hybrid_equity_pct)
    inv_cf_hyb_debt_disp = convert_series(inv_cf_hyb_debt, CUR, fx)

    invested_h = -min(inv_cf_hyb_debt_disp.iloc[0], 0.0)
    returned_h = inv_cf_hyb_debt_disp[inv_cf_hyb_debt_disp > 0].sum()
    two_line_card("How much you invest today (Debt leg)", fmt_money(invested_h, CUR), "Loan principal at t0")
    two_line_card("How much you will get back (Debt leg)", fmt_money(returned_h, CUR), "Interest + principal")
    info_card("When", "Interest 2028–2030 • Bullet 2030")

    st.markdown("#### Company cashflows")
    st.dataframe(cf_hyb.to_frame(f"Company Net CF ({CUR})").style.format("{:,.0f}"))

    fig, ax = plt.subplots()
    ax.bar(cf_hyb.index.astype(str), cf_hyb.values)
    ax.set_title(f"Company Net Cashflows ({CUR})"); ax.set_xlabel("Year"); ax.set_ylabel(CUR)
    st.pyplot(fig)

# ---------- Founder tab ----------
with tab_founder:
    st.subheader("Founder — Cash by year, bullet coverage, next raise timing & size")

    scen_choice = st.selectbox("Scenario to evaluate", ["Equity", "Debt", "Hybrid"], index=1)

    if scen_choice == "Equity":
        cf_use_aed = company_net_cf_equity_aed(ask_aed)
        end_use_aed = cf_use_aed.cumsum() + start_cash_aed
        bullet_year = None
        bullet_amt_aed = 0.0
    elif scen_choice == "Debt":
        cf_use_aed, bullet_amt_aed = company_net_cf_debt_aed(ask_aed)
        end_use_aed = cf_use_aed.cumsum() + start_cash_aed
        bullet_year = int(YEARS[-1])  # 2030
    else:
        cf_use_aed, bullet_amt_aed = company_net_cf_hybrid_aed(ask_aed, hybrid_equity_pct)
        end_use_aed = cf_use_aed.cumsum() + start_cash_aed
        bullet_year = int(YEARS[-1])  # 2030

    # Display-currency series
    cf_use = convert_series(cf_use_aed, CUR, fx)
    end_use = convert_series(end_use_aed, CUR, fx)
    bullet_amt = convert_amount(bullet_amt_aed, CUR, fx)

    # Clear metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        two_line_card("Starting cash", fmt_money(start_cash_display, CUR),
                      f"Min buffer: {fmt_money(buffer_display, CUR)}")
    with c2:
        two_line_card("Cash at end (first year)", fmt_money(end_use.iloc[0], CUR), f"Year {end_use.index[0]}")
    with c3:
        two_line_card("Cash at end (last year)", fmt_money(end_use.iloc[-1], CUR), f"Year {end_use.index[-1]}")

    # Bullet coverage snapshot
    if bullet_year is None:
        info_card("Bullet coverage", "Not applicable")
    else:
        cash_by_bullet = float(end_use.loc[bullet_year])
        status = "Yes" if cash_by_bullet >= bullet_amt else "No"
        two_line_card("Bullet coverage", status,
                      f"Cash @ {bullet_year}: {fmt_money(cash_by_bullet, CUR)} • Bullet: {fmt_money(bullet_amt, CUR)}")

    st.markdown("#### Ending cash by year")
    df_founder = pd.DataFrame({
        "Year": end_use.index,
        f"Company Net CF ({CUR})": cf_use.values,
        f"Ending Cash ({CUR})": end_use.values
    }).set_index("Year")
    st.dataframe(df_founder.applymap(lambda x: fmt_int(x)))

    fig1, ax1 = plt.subplots()
    ax1.plot(end_use.index, end_use.values, marker="o")
    ax1.axhline(buffer_display, linestyle="--", linewidth=1)
    ax1.set_title(f"Ending Cash vs Buffer ({CUR})")
    ax1.set_xlabel("Year"); ax1.set_ylabel(CUR)
    st.pyplot(fig1)

    # -----------------------------
    # FUTURE RAISE RECOMMENDATION (WHEN & HOW MUCH)
    # -----------------------------
    st.markdown("#### Future raise recommendation (for bullet coverage & positive cashflow)")

    # 1) First year where ending cash dips below buffer
    shortfall_year = None
    for yr in end_use_aed.index:
        if end_use_aed.loc[yr] < buffer_aed:
            shortfall_year = int(yr)
            break

    if shortfall_year is None:
        st.success("✅ You are self-sustained through 2030 under current funding and assumptions.")
    else:
        # 2) Estimate next 12-month need using the following year's base NCF magnitude as proxy
        # If we are at 2028, use 2029 BASE_NCF as 12-month need; if not available, use current year's |BASE_NCF|
        next_year = shortfall_year + 1
        if next_year in BASE_NCF_AED.index:
            next_year_expense = abs(BASE_NCF_AED.loc[next_year])
        else:
            next_year_expense = abs(BASE_NCF_AED.loc[shortfall_year])

        # 3) Bullet alignment: if bullet is in shortfall year or the next, include it
        bullet_due = (bullet_year is not None) and (bullet_year in [shortfall_year, shortfall_year + 1])
        bullet_add_aed = bullet_amt_aed if bullet_due else 0.0

        # 4) Amount to raise in the shortfall year:
        #    raise = (buffer target - projected end cash) + 12-month expense + bullet if due
        needed_raise_aed = max(0.0, (buffer_aed - end_use_aed.loc[shortfall_year]) + next_year_expense + bullet_add_aed)
        needed_raise_disp = convert_amount(needed_raise_aed, CUR, fx)

        when_text = f"{shortfall_year} (before cash dips below buffer)"
        if bullet_add_aed > 0:
            when_text += f" — also aligns with {bullet_year} bullet repayment"

        st.warning(f"⚠️ Expected cash shortfall in **{shortfall_year}**.")
        two_line_card("Ideal time for next raise", when_text, "To restore 12-month runway & cover obligations")
        two_line_card("Recommended raise amount", fmt_money(needed_raise_disp, CUR),
                      f"Includes bullet coverage: {fmt_money(convert_amount(bullet_add_aed, CUR, fx), CUR)}")

# ---------- Compare tab ----------
with tab_compare:
    st.subheader("Quick compare (company net cashflows)")
    fig2, ax2 = plt.subplots()
    ax2.plot(cf_equity.index, cf_equity.values, marker="o", label="Equity")
    ax2.plot(cf_debt.index,   cf_debt.values,   marker="o", label="Debt")
    ax2.plot(cf_hyb.index,    cf_hyb.values,    marker="o", label="Hybrid")
    ax2.set_title(f"Annual Company Net Cashflows ({CUR})")
    ax2.set_xlabel("Year"); ax2.set_ylabel(CUR); ax2.legend()
    st.pyplot(fig2)
