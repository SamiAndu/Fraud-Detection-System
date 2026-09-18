import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0b1120;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .risk-high {
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #ef4444;
        background: rgba(239, 68, 68, 0.12);
        color: #fca5a5;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }

    .risk-medium {
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #f59e0b;
        background: rgba(245, 158, 11, 0.12);
        color: #fcd34d;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }

    .risk-low {
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #22c55e;
        background: rgba(34, 197, 94, 0.12);
        color: #86efac;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

DATA_PATH = "data/processed/final_risk_results.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "Fraud results were not found. "
        "Run `python src/scoring.py` first."
    )
    st.stop()


# ============================================================
# DERIVED METRICS
# ============================================================

total_transactions = len(df)

high_risk = (
        df["final_risk_level"] == "HIGH"
).sum()

medium_risk = (
        df["final_risk_level"] == "MEDIUM"
).sum()

low_risk = (
        df["final_risk_level"] == "LOW"
).sum()

high_risk_rate = (
        high_risk / total_transactions * 100
)

avg_ml_probability = (
    df["ml_fraud_probability_pct"].mean()
)

avg_hybrid_score = (
    df["hybrid_risk_score"].mean()
)

total_amount = (
    df["amount"].sum()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ Fraud Engine")

    st.caption(
        "Hybrid Rules + Machine Learning"
    )

    st.divider()

    st.markdown("### Filters")

    selected_risk = st.multiselect(
        "Risk Level",
        options=["LOW", "MEDIUM", "HIGH"],
        default=["LOW", "MEDIUM", "HIGH"],
    )

    selected_country = st.multiselect(
        "Country",
        options=sorted(df["country"].unique()),
        default=[],
    )

    if selected_risk:
        filtered_df = df[
            df["final_risk_level"].isin(selected_risk)
        ]
    else:
        filtered_df = df.copy()

    if selected_country:
        filtered_df = filtered_df[
            filtered_df["country"].isin(selected_country)
        ]

    st.divider()

    st.markdown("### System")

    st.caption("Data source")
    st.write("Synthetic transaction dataset")

    st.caption("Detection approach")
    st.write("Rules + Random Forest")

    st.caption("Transactions")
    st.write(f"{total_transactions:,}")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    '🛡️ Fraud Detection Engine'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Real-time-style transaction risk analysis using '
    'explainable rules and machine learning.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# KPI ROW
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Transactions",
        f"{len(filtered_df):,}",
    )

with col2:
    st.metric(
        "High Risk",
        f"{high_risk:,}",
    )

with col3:
    st.metric(
        "High Risk Rate",
        f"{high_risk_rate:.2f}%",
    )

with col4:
    st.metric(
        "Avg ML Probability",
        f"{avg_ml_probability:.2f}%",
    )

with col5:
    st.metric(
        "Avg Hybrid Score",
        f"{avg_hybrid_score:.2f}",
    )


st.divider()


# ============================================================
# ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Risk Analytics'
    '</div>',
    unsafe_allow_html=True,
)

chart_col1, chart_col2 = st.columns(2)


with chart_col1:

    st.markdown("#### Risk Distribution")

risk_counts = (
    filtered_df["final_risk_level"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH"],
        fill_value=0,
    )
)

risk_chart_df = pd.DataFrame({
    "Risk Level": risk_counts.index,
    "Transactions": risk_counts.values,
})

st.bar_chart(
    risk_chart_df,
    x="Risk Level",
    y="Transactions",
)


with chart_col2:

    st.markdown("#### Transactions by Country")

    country_counts = (
        filtered_df["country"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(country_counts)


# ============================================================
# TRANSACTION INVESTIGATOR
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Transaction Investigator'
    '</div>',
    unsafe_allow_html=True,
)

if len(filtered_df) == 0:

    st.warning(
        "No transactions match the selected filters."
    )

else:

    transaction_ids = (
        filtered_df["transaction_id"]
        .tolist()
    )

    transaction_id = st.selectbox(
        "Select a transaction to investigate",
        transaction_ids,
    )

    transaction = df[
        df["transaction_id"] == transaction_id
        ].iloc[0]


    # --------------------------------------------------------
    # Transaction details
    # --------------------------------------------------------

    info1, info2, info3 = st.columns(3)


    # --------------------------------------------------------
    # Transaction
    # --------------------------------------------------------

    with info1:

        st.markdown("#### Transaction")

        st.write(
            f"**Transaction ID:** "
            f"{transaction['transaction_id']}"
        )

        if "customer_id" in transaction.index:
            st.write(
                f"**Customer ID:** "
                f"{transaction['customer_id']}"
            )

        if "amount" in transaction.index:
            st.write(
                f"**Amount:** "
                f"${transaction['amount']:,.2f}"
            )

        if "merchant_category" in transaction.index:
            st.write(
                f"**Merchant:** "
                f"{transaction['merchant_category']}"
            )


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    with info2:

        st.markdown("#### Context")

        if "country" in transaction.index:
            st.write(
                f"**Country:** "
                f"{transaction['country']}"
            )

        if "transaction_hour" in transaction.index:
            st.write(
                f"**Transaction Hour:** "
                f"{transaction['transaction_hour']}:00"
            )

        if "previous_transaction_minutes" in transaction.index:
            st.write(
                f"**Previous Transaction:** "
                f"{transaction['previous_transaction_minutes']} min ago"
            )

        if "transactions_last_24h" in transaction.index:
            st.write(
                f"**Transactions / 24h:** "
                f"{transaction['transactions_last_24h']}"
            )


    # --------------------------------------------------------
    # Risk Metrics
    # --------------------------------------------------------

    with info3:

        st.markdown("#### Risk Analysis")

        if "rule_risk_score" in transaction.index:
            st.write(
                f"**Rule Score:** "
                f"{transaction['rule_risk_score']}"
            )

        if "ml_fraud_probability_pct" in transaction.index:
            st.write(
                f"**ML Probability:** "
                f"{transaction['ml_fraud_probability_pct']:.2f}%"
            )

        if "hybrid_risk_score" in transaction.index:
            st.write(
                f"**Hybrid Score:** "
                f"{transaction['hybrid_risk_score']:.2f}"
            )

        if "final_risk_level" in transaction.index:
            st.write(
                f"**Final Risk:** "
                f"**{transaction['final_risk_level']}**"
            )


    # --------------------------------------------------------
    # Final risk assessment
    # --------------------------------------------------------

    risk_level = transaction["final_risk_level"]


    if risk_level == "HIGH":

        st.markdown(
            '<div class="risk-high">'
            '🚨 HIGH RISK TRANSACTION'
            '</div>',
            unsafe_allow_html=True,
        )

        st.error(
            "Assessment: This transaction should be investigated "
            "before approval."
        )

    elif risk_level == "MEDIUM":

        st.markdown(
            '<div class="risk-medium">'
            '⚠️ MEDIUM RISK TRANSACTION'
            '</div>',
            unsafe_allow_html=True,
        )

        st.warning(
            "Assessment: This transaction warrants additional "
            "review."
        )

    else:

        st.markdown(
            '<div class="risk-low">'
            '✓ LOW RISK TRANSACTION'
            '</div>',
            unsafe_allow_html=True,
        )

        st.success(
            "Assessment: No immediate escalation is indicated."
        )


    # --------------------------------------------------------
    # Detection reasons
    # --------------------------------------------------------

    st.markdown("### Detection Reasons")

    if (
            "rule_reasons" in transaction.index
            and pd.notna(transaction["rule_reasons"])
            and str(transaction["rule_reasons"]).strip() != "None"
    ):

        reasons = str(
            transaction["rule_reasons"]
        ).split(";")

        for reason in reasons:

            reason = reason.strip()

            if reason:
                st.markdown(
                    f"✓ **{reason}**"
                )

    else:

        st.info(
            "No rule-based detection reasons were recorded "
            "for this transaction."
        )


    # --------------------------------------------------------
    # Detection reasons
    # --------------------------------------------------------

    st.markdown("#### Detection Reasons")

    reasons = transaction["rule_reasons"]


    if pd.isna(reasons) or str(reasons).strip() == "":

        st.info(
            "No rule-based risk factors were triggered."
        )

    else:

        reasons_list = str(reasons).split(";")

        for reason in reasons_list:

            reason = reason.strip()

            if reason:
                st.write(f"🔹 {reason}")


# ============================================================
# HIGH-RISK TRANSACTIONS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'High-Risk Transactions'
    '</div>',
    unsafe_allow_html=True,
)

high_risk_df = (
    df[
        df["final_risk_level"] == "HIGH"
        ]
    .sort_values(
        "hybrid_risk_score",
        ascending=False,
    )
)


display_columns = [
    "transaction_id",
    "amount",
    "country",
    "rule_risk_score",
    "ml_fraud_probability_pct",
    "hybrid_risk_score",
    "final_risk_level",
    "rule_reasons",
]


st.dataframe(
    high_risk_df[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config={
        "transaction_id": st.column_config.TextColumn(
            "Transaction"
        ),

        "amount": st.column_config.NumberColumn(
            "Amount",
            format="$%.2f",
        ),

        "rule_risk_score": st.column_config.NumberColumn(
            "Rule Score"
        ),

        "ml_fraud_probability_pct": st.column_config.NumberColumn(
            "ML Probability",
            format="%.2f%%",
        ),

        "hybrid_risk_score": st.column_config.NumberColumn(
            "Hybrid Score",
            format="%.2f",
        ),

        "final_risk_level": st.column_config.TextColumn(
            "Risk Level"
        ),

        "rule_reasons": st.column_config.TextColumn(
            "Detection Reasons"
        ),
    },
)


# ============================================================
# DATASET SUMMARY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Dataset Summary'
    '</div>',
    unsafe_allow_html=True,
)

summary1, summary2, summary3 = st.columns(3)

with summary1:

    st.metric(
        "Total Transaction Value",
        f"${total_amount:,.2f}",
    )

with summary2:

    st.metric(
        "Medium Risk Transactions",
        f"{medium_risk:,}",
    )

with summary3:

    st.metric(
        "Low Risk Transactions",
        f"{low_risk:,}",
    )


# ============================================================
# FULL DATASET
# ============================================================

with st.expander("View Full Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fraud Detection Engine • Synthetic data • "
    "Rules + Machine Learning"
)