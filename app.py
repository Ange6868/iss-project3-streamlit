import streamlit as st
import pandas as pd
import numpy as np
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

@st.cache_data
def load_forecast_data():
    try:
        forecast_df = pd.read_csv("fas_forecast_data.csv")
        forecast_df["year"] = pd.to_numeric(forecast_df["year"], errors="coerce")
        forecast_df["value"] = pd.to_numeric(forecast_df["value"], errors="coerce")
        forecast_df = forecast_df.dropna(subset=["year", "value"]).copy()
        forecast_df["year"] = forecast_df["year"].astype(int)
        return forecast_df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()
forecast_df = load_forecast_data()

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
# Predictive outlook
# =========================

st.header("6. Predictive Outlook")

st.markdown(
    """
    This section uses historical IMF Financial Access Survey data to create a simple
    regression-based projection for selected financial access indicators.
    """
)

if forecast_df.empty:
    st.info(
        "Forecast data is not available yet. Please upload fas_forecast_data.csv to enable this section."
    )
else:
    forecast_countries = sorted(forecast_df["REF_AREA_LABEL"].dropna().unique().tolist())
    forecast_indicators = sorted(forecast_df["short_indicator"].dropna().unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        forecast_country = st.selectbox(
            "Select a country/economy for projection",
            forecast_countries,
            key="forecast_country"
        )

    with col2:
        forecast_indicator = st.selectbox(
            "Select a financial access indicator for projection",
            forecast_indicators,
            format_func=pretty_name,
            key="forecast_indicator"
        )

    model_data = forecast_df[
        forecast_df["short_indicator"] == forecast_indicator
    ].copy()

    selected_hist = model_data[
        model_data["REF_AREA_LABEL"] == forecast_country
    ].sort_values("year").copy()

    if selected_hist.shape[0] < 3:
        st.warning(
            "Not enough historical observations for the selected country and indicator."
        )
    else:
        # Use a pooled regression with country fixed effects:
        # value = intercept + year trend + country baseline adjustment
        # This uses all selected countries' historical data for the same indicator.
        model_data = model_data.dropna(subset=["REF_AREA_LABEL", "year", "value"]).copy()
        model_data["year_centered"] = model_data["year"] - model_data["year"].min()

        country_dummies = pd.get_dummies(
            model_data["REF_AREA_LABEL"],
            prefix="country",
            drop_first=True,
            dtype=float
        )

        X = pd.concat(
            [
                pd.Series(1.0, index=model_data.index, name="intercept"),
                model_data[["year_centered"]],
                country_dummies
            ],
            axis=1
        )

        y = model_data["value"].astype(float)

        coef = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        model_data["predicted"] = X.values @ coef

        # Model fit, used only as a rough diagnostic
        ss_res = ((model_data["value"] - model_data["predicted"]) ** 2).sum()
        ss_tot = ((model_data["value"] - model_data["value"].mean()) ** 2).sum()
        r_squared = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

        last_year = int(selected_hist["year"].max())
        future_years = list(range(last_year + 1, last_year + 6))

        future_df = pd.DataFrame({
            "REF_AREA_LABEL": forecast_country,
            "year": future_years
        })

        future_df["year_centered"] = future_df["year"] - model_data["year"].min()

        # Build future design matrix with same columns as training X
        X_future = pd.DataFrame(0.0, index=future_df.index, columns=X.columns)
        X_future["intercept"] = 1.0
        X_future["year_centered"] = future_df["year_centered"]

        country_col_name = f"country_{forecast_country}"
        if country_col_name in X_future.columns:
            X_future[country_col_name] = 1.0

        future_df["value"] = X_future.values @ coef
        future_df["value"] = future_df["value"].clip(lower=0)

        # Prepare plot data
        actual_plot = selected_hist[["year", "value"]].copy()
        actual_plot["series"] = "Historical"

        last_actual = selected_hist[["year", "value"]].tail(1).copy()
        last_actual["series"] = "Projected"

        projected_plot = future_df[["year", "value"]].copy()
        projected_plot["series"] = "Projected"

        plot_df = pd.concat(
            [actual_plot, last_actual, projected_plot],
            ignore_index=True
        )

        fig = px.line(
            plot_df,
            x="year",
            y="value",
            color="series",
            markers=True,
            title=f"{pretty_name(forecast_indicator)}: Historical Trend and 5-Year Projection",
            labels={
                "year": "Year",
                "value": pretty_name(forecast_indicator),
                "series": "Series"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        metric1, metric2, metric3 = st.columns(3)

        current_value = selected_hist["value"].iloc[-1]
        projected_value = future_df["value"].iloc[-1]
        projected_change = projected_value - current_value

        metric1.metric(
            "Latest historical value",
            f"{current_value:.1f}"
        )

        metric2.metric(
            "Projected value in 5 years",
            f"{projected_value:.1f}"
        )

        metric3.metric(
            "Projected change",
            f"{projected_change:+.1f}"
        )

        st.write(f"Model R-squared: **{r_squared:.2f}**")

        st.dataframe(
            pd.concat(
                [
                    selected_hist[["REF_AREA_LABEL", "short_indicator", "year", "value"]],
                    future_df.assign(short_indicator=forecast_indicator)[
                        ["REF_AREA_LABEL", "short_indicator", "year", "value"]
                    ]
                ],
                ignore_index=True
            ),
            use_container_width=True
        )

        st.caption(
            "This projection uses a simple pooled regression model with a year trend and country-level baseline differences. "
            "It is intended as an exploratory scenario, not a precise forecast. Results should be interpreted carefully, "
            "especially because the number of countries and historical observations is limited."
        )

# =========================
# Data table
# =========================

st.header("7. Final Analysis Data")

with st.expander("Show final dataset"):
    st.dataframe(df, use_container_width=True)

# =========================
# Data notes
# =========================

st.header("8. Data Notes")

st.markdown(
    """
    - Education indicators are based on mathematics-related parity measures from the education dataset.
    - Financial usage indicators are based on World Bank Findex data.
    - Financial access infrastructure indicators are based on IMF Financial Access Survey data.
    - The analysis uses nearest available years across datasets when exact-year alignment is not available.
    - Results should be interpreted as exploratory comparisons rather than causal evidence.
    """
)
