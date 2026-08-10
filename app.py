import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# Page setup
# =========================

st.set_page_config(
    page_title="SDG Education & Financial Inclusion Explorer",
    page_icon="📊",
    layout="wide"
)

# =========================
# Load data
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("project3_analysis_data.csv")
    return df

df = load_data()

# =========================
# Helper lists
# =========================

country_col = "REF_AREA_LABEL"

score_cols = [
    col for col in [
        "education_equity_score",
        "financial_usage_score",
        "financial_access_score",
        "overall_readiness_score"
    ]
    if col in df.columns
]

education_cols = [
    col for col in [
        "math_gpi",
        "math_wpi",
        "math_test_language_pi",
        "math_location_pi",
        "math_native_pi"
    ]
    if col in df.columns
]

financial_usage_cols = [
    col for col in [
        "account_ownership",
        "financial_institution_account",
        "digital_payment",
        "used_debit_card",
        "owns_credit_card",
        "formal_saving",
        "formal_borrowing"
    ]
    if col in df.columns
]

financial_access_cols = [
    col for col in [
        "atm_per_100k_adults",
        "bank_branches_per_100k_adults",
        "deposit_accounts_per_1000_adults",
        "credit_cards_per_1000_adults",
        "debit_cards_per_1000_adults"
    ]
    if col in df.columns
]

indicator_label_map = {
    "math_gpi": "Math gender parity index",
    "math_wpi": "Math wealth parity index",
    "math_test_language_pi": "Math test language parity index",
    "math_location_pi": "Math location parity index",
    "math_native_pi": "Math native parity index",
    "account_ownership": "Account ownership",
    "financial_institution_account": "Financial institution account",
    "digital_payment": "Made or received a digital payment",
    "used_debit_card": "Used a debit card",
    "owns_credit_card": "Owns a credit card",
    "formal_saving": "Saved at a financial institution",
    "formal_borrowing": "Borrowed from a financial institution",
    "atm_per_100k_adults": "ATMs per 100,000 adults",
    "bank_branches_per_100k_adults": "Bank branches per 100,000 adults",
    "deposit_accounts_per_1000_adults": "Deposit accounts per 1,000 adults",
    "credit_cards_per_1000_adults": "Credit cards per 1,000 adults",
    "debit_cards_per_1000_adults": "Debit cards per 1,000 adults",
    "education_equity_score": "Education Equity Score",
    "financial_usage_score": "Financial Usage Score",
    "financial_access_score": "Financial Access Score",
    "overall_readiness_score": "Overall Readiness Score"
}

def pretty_name(col):
    return indicator_label_map.get(col, col.replace("_", " ").title())

# =========================
# Title
# =========================

st.title("SDG Education & Financial Inclusion Explorer")

st.markdown(
    """
    This app explores mathematics-related education equity and financial inclusion readiness
    across selected Asian economies. It combines SDG 4 education-related indicators with
    SDG 8.10 financial inclusion indicators.
    """
)

# =========================
# Sidebar
# =========================

st.sidebar.header("Controls")

countries = df[country_col].dropna().unique().tolist()
selected_country = st.sidebar.selectbox("Select a country/economy", countries)

selected_row = df[df[country_col] == selected_country].iloc[0]

# =========================
# Overview metrics
# =========================

st.header("1. Country Profile")

metric_cols = st.columns(4)

if "overall_readiness_score" in df.columns:
    metric_cols[0].metric(
        "Overall Readiness Score",
        f"{selected_row['overall_readiness_score']:.1f}"
    )

if "education_equity_score" in df.columns:
    metric_cols[1].metric(
        "Education Equity Score",
        f"{selected_row['education_equity_score']:.1f}"
    )

if "financial_usage_score" in df.columns:
    metric_cols[2].metric(
        "Financial Usage Score",
        f"{selected_row['financial_usage_score']:.1f}"
    )

if "financial_access_score" in df.columns:
    metric_cols[3].metric(
        "Financial Access Score",
        f"{selected_row['financial_access_score']:.1f}"
    )

st.subheader("Data years used")

year_cols = [col for col in ["edu_year", "findex_year", "fas_year"] if col in df.columns]

if year_cols:
    st.dataframe(
        df[[country_col] + year_cols],
        use_container_width=True
    )

# =========================
# Composite score ranking
# =========================

st.header("2. Composite Score Ranking")

if "overall_readiness_score" in df.columns:
    ranking_df = df.sort_values("overall_readiness_score", ascending=False).copy()
    
    fig = px.bar(
        ranking_df,
        x=country_col,
        y="overall_readiness_score",
        text="overall_readiness_score",
        title="Overall Readiness Score by Country/Economy",
        labels={
            country_col: "Country/Economy",
            "overall_readiness_score": "Overall Readiness Score"
        }
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)

    show_cols = [country_col] + score_cols
    if "rank" in df.columns:
        show_cols = ["rank"] + show_cols

    st.dataframe(
        ranking_df[show_cols],
        use_container_width=True
    )
else:
    st.info("Overall readiness score is not available in the dataset.")

# =========================
# Education equity
# =========================

st.header("3. Education Equity Indicators")

if education_cols:
    selected_edu_indicator = st.selectbox(
        "Select an education indicator",
        education_cols,
        format_func=pretty_name
    )

    fig = px.bar(
        df,
        x=country_col,
        y=selected_edu_indicator,
        text=selected_edu_indicator,
        title=pretty_name(selected_edu_indicator),
        labels={
            country_col: "Country/Economy",
            selected_edu_indicator: pretty_name(selected_edu_indicator)
        }
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "For parity indicators, values closer to 1 generally suggest more equal outcomes across groups."
    )
else:
    st.info("No education equity indicators were found in the dataset.")

# =========================
# Financial inclusion
# =========================

st.header("4. Financial Inclusion Indicators")

financial_cols = financial_usage_cols + financial_access_cols

if financial_cols:
    selected_fin_indicator = st.selectbox(
        "Select a financial inclusion indicator",
        financial_cols,
        format_func=pretty_name
    )

    fig = px.bar(
        df,
        x=country_col,
        y=selected_fin_indicator,
        text=selected_fin_indicator,
        title=pretty_name(selected_fin_indicator),
        labels={
            country_col: "Country/Economy",
            selected_fin_indicator: pretty_name(selected_fin_indicator)
        }
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[[country_col, selected_fin_indicator]],
        use_container_width=True
    )
else:
    st.info("No financial inclusion indicators were found in the dataset.")

# =========================
# Cross-SDG comparison
# =========================

st.header("5. Cross-SDG Comparison")

if education_cols and financial_cols:
    col1, col2 = st.columns(2)

    with col1:
        x_indicator = st.selectbox(
            "X-axis: education indicator",
            education_cols,
            format_func=pretty_name
        )

    with col2:
        y_indicator = st.selectbox(
            "Y-axis: financial inclusion indicator",
            financial_cols,
            format_func=pretty_name
        )

    fig = px.scatter(
        df,
        x=x_indicator,
        y=y_indicator,
        text=country_col,
        title=f"{pretty_name(x_indicator)} vs. {pretty_name(y_indicator)}",
        labels={
            x_indicator: pretty_name(x_indicator),
            y_indicator: pretty_name(y_indicator)
        }
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    corr_df = df[[x_indicator, y_indicator]].dropna()

    if len(corr_df) >= 3:
        corr = corr_df[x_indicator].corr(corr_df[y_indicator])
        st.write(f"Exploratory correlation: **{corr:.2f}**")
    else:
        st.write("Not enough non-missing observations to calculate a correlation.")

    st.caption(
        "Because this comparison includes only a small number of countries/economies, "
        "the scatter plot should be interpreted as exploratory rather than causal or statistically conclusive."
    )
else:
    st.info("Cross-SDG comparison requires both education and financial indicators.")

# =========================
# Data table
# =========================

st.header("6. Final Analysis Data")

with st.expander("Show final dataset"):
    st.dataframe(df, use_container_width=True)

# =========================
# Data notes
# =========================

st.header("7. Data Notes")

st.markdown(
    """
    - Education indicators are based on mathematics-related parity measures from the education dataset.
    - Financial usage indicators are based on World Bank Findex data.
    - Financial access infrastructure indicators are based on IMF Financial Access Survey data.
    - The analysis uses nearest available years across datasets when exact-year alignment is not available.
    - Results should be interpreted as exploratory comparisons rather than causal evidence.
    """
)
