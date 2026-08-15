import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="SDG 4 & 8 Explorer",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Global settings
# ============================================================

DATA_PATH = Path("project3_analysis_data.csv")
FORECAST_PATH = Path("fas_forecast_data.csv")

COUNTRY_ORDER = [
    "Korea, Rep.",
    "Hong Kong SAR, China",
    "Thailand",
    "Indonesia"
]

COUNTRY_LABEL_MAP = {
    "Korea, Rep.": "Korea",
    "Hong Kong SAR, China": "Hong Kong",
    "Thailand": "Thailand",
    "Indonesia": "Indonesia"
}

COUNTRY_LABEL_ORDER = [
    "Korea",
    "Hong Kong",
    "Thailand",
    "Indonesia"
]

COUNTRY_COLOR_MAP = {
    "Korea, Rep.": "#FF6347",
    "Hong Kong SAR, China": "#FA8072",
    "Thailand": "#FF8C00",
    "Indonesia": "#FFE34D"
}

PILLAR_ORDER = [
    "Mathematics learning equity",
    "Formal financial participation",
    "Financial access infrastructure"
]

PILLAR_COLOR_MAP = {
    "Mathematics learning equity": "#66CDAA",
    "Formal financial participation": "#20B2AA",
    "Financial access infrastructure": "#B0E0E6"
}

EDUCATION_RAW_COLS = [
    "math_gpi",
    "math_wpi",
    "math_test_language_pi",
    "math_location_pi",
    "math_native_pi"
]

FINDEX_RAW_COLS = [
    "account_ownership",
    "financial_institution_account",
    "digital_payment",
    "used_debit_card",
    "owns_credit_card",
    "formal_saving",
    "formal_borrowing"
]

FAS_RAW_COLS = [
    "atm_per_100k_adults",
    "bank_branches_per_100k_adults",
    "deposit_accounts_per_1000_adults",
    "credit_cards_per_1000_adults",
    "debit_cards_per_1000_adults"
]

INDICATOR_LABEL_MAP = {
    # Education
    "math_gpi": "Math gender parity index",
    "math_wpi": "Math wealth parity index",
    "math_test_language_pi": "Math test-language parity index",
    "math_location_pi": "Math location parity index",
    "math_native_pi": "Math native/background parity index",

    # Findex
    "account_ownership": "Account ownership",
    "financial_institution_account": "Financial institution account",
    "digital_payment": "Made or received a digital payment",
    "used_debit_card": "Used a debit card",
    "owns_credit_card": "Owns a credit card",
    "formal_saving": "Saved at a financial institution",
    "formal_borrowing": "Borrowed from a financial institution",

    # FAS
    "atm_per_100k_adults": "ATMs per 100,000 adults",
    "bank_branches_per_100k_adults": "Commercial bank branches per 100,000 adults",
    "deposit_accounts_per_1000_adults": "Deposit accounts per 1,000 adults",
    "credit_cards_per_1000_adults": "Credit cards per 1,000 adults",
    "debit_cards_per_1000_adults": "Debit cards per 1,000 adults",

    # Scores
    "education_equity_score": "Mathematics learning equity score",
    "formal_financial_participation_score": "Formal financial participation score",
    "financial_access_infrastructure_score": "Financial access infrastructure score",
    "financial_inclusion_ecosystem_score": "Financial inclusion ecosystem score",
    "overall_readiness_score": "Overall readiness score"
}


# ============================================================
# Helper functions
# ============================================================

def minmax_score(series: pd.Series) -> pd.Series:
    """
    Convert a numeric indicator to a 0-100 relative score across selected economies.
    This is used for FAS infrastructure indicators because they have different units.
    """
    series = pd.to_numeric(series, errors="coerce")
    valid = series.dropna()

    if valid.empty or valid.max() == valid.min():
        return pd.Series(np.nan, index=series.index)

    return (series - valid.min()) / (valid.max() - valid.min()) * 100


def percent_score(series: pd.Series) -> pd.Series:
    """
    Keep percentage-based indicators on their original 0-100 scale.
    If a column is stored as a proportion between 0 and 1, convert it to 0-100.
    This is used for Findex indicators because they already represent usage percentages.
    """
    series = pd.to_numeric(series, errors="coerce")
    valid = series.dropna()

    if not valid.empty and valid.max() <= 1.5:
        series = series * 100

    return series.clip(lower=0, upper=100)


def pretty_name(col_name: str) -> str:
    return INDICATOR_LABEL_MAP.get(
        col_name,
        col_name.replace("_", " ").title()
    )


def add_country_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "REF_AREA_LABEL" in df.columns:
        df["economy_label"] = df["REF_AREA_LABEL"].map(COUNTRY_LABEL_MAP).fillna(df["REF_AREA_LABEL"])
    return df


def format_score(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}"


def ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute score columns using the updated project logic:
    - Education parity score: closer to 1 = more equitable
    - Findex participation score: average of percentage-based usage indicators
    - FAS infrastructure score: min-max normalized because units differ
    """
    df = df.copy()

    education_equity_cols = [col for col in EDUCATION_RAW_COLS if col in df.columns]
    financial_participation_cols = [col for col in FINDEX_RAW_COLS if col in df.columns]
    financial_access_cols = [col for col in FAS_RAW_COLS if col in df.columns]

    education_score_cols = []
    for col in education_equity_cols:
        score_col = col + "_score"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[score_col] = (100 - (df[col] - 1).abs() * 100).clip(lower=0, upper=100)
        education_score_cols.append(score_col)

    participation_score_cols = []
    for col in financial_participation_cols:
        score_col = col + "_score"
        df[score_col] = percent_score(df[col])
        participation_score_cols.append(score_col)

    access_score_cols = []
    for col in financial_access_cols:
        score_col = col + "_score"
        df[score_col] = minmax_score(df[col])
        access_score_cols.append(score_col)

    if education_score_cols:
        df["education_equity_score"] = df[education_score_cols].mean(axis=1)

    if participation_score_cols:
        df["formal_financial_participation_score"] = df[participation_score_cols].mean(axis=1)

    if access_score_cols:
        df["financial_access_infrastructure_score"] = df[access_score_cols].mean(axis=1)

    required_pillars = [
        "education_equity_score",
        "formal_financial_participation_score",
        "financial_access_infrastructure_score"
    ]

    if all(col in df.columns for col in required_pillars):
        df["financial_inclusion_ecosystem_score"] = df[required_pillars].mean(axis=1)

    if "formal_financial_participation_score" in df.columns:
        df["financial_usage_score"] = df["formal_financial_participation_score"]

    if "financial_access_infrastructure_score" in df.columns:
        df["financial_access_score"] = df["financial_access_infrastructure_score"]

    if "financial_inclusion_ecosystem_score" in df.columns:
        df["overall_readiness_score"] = df["financial_inclusion_ecosystem_score"]

        df = df.sort_values(
            "financial_inclusion_ecosystem_score",
            ascending=False
        ).reset_index(drop=True)

        df["rank"] = df.index + 1

    df = add_country_labels(df)
    return df


@st.cache_data
def load_analysis_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            "Could not find project3_analysis_data.csv. "
            "Please upload it to the same folder as app.py."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df = ensure_scores(df)
    return df


@st.cache_data
def load_forecast_data() -> pd.DataFrame:
    if not FORECAST_PATH.exists():
        return pd.DataFrame()

    forecast_df = pd.read_csv(FORECAST_PATH)

    forecast_df["year"] = pd.to_numeric(forecast_df["year"], errors="coerce")
    forecast_df["value"] = pd.to_numeric(forecast_df["value"], errors="coerce")

    forecast_df = forecast_df.dropna(subset=["year", "value"]).copy()
    forecast_df["year"] = forecast_df["year"].astype(int)

    forecast_df = add_country_labels(forecast_df)
    return forecast_df


# ============================================================
# Plot functions
# ============================================================

def plot_ecosystem_score(df: pd.DataFrame):
    plot_df = df.sort_values("financial_inclusion_ecosystem_score", ascending=False)

    fig = px.bar(
        plot_df,
        x="economy_label",
        y="financial_inclusion_ecosystem_score",
        color="REF_AREA_LABEL",
        text="financial_inclusion_ecosystem_score",
        title="Financial Inclusion Ecosystem Score by Economy",
        labels={
            "economy_label": "Economy",
            "financial_inclusion_ecosystem_score": "Financial Inclusion Ecosystem Score",
            "REF_AREA_LABEL": "Economy"
        },
        color_discrete_map=COUNTRY_COLOR_MAP,
        category_orders={
            "economy_label": COUNTRY_LABEL_ORDER,
            "REF_AREA_LABEL": COUNTRY_ORDER
        }
    )

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis_range=[0, 105]
    )

    return fig


def plot_three_dimension_profile(df: pd.DataFrame):
    pillar_score_map = {
        "education_equity_score": "Mathematics learning equity",
        "formal_financial_participation_score": "Formal financial participation",
        "financial_access_infrastructure_score": "Financial access infrastructure"
    }

    pillar_plot_df = df[
        ["REF_AREA_LABEL", "economy_label"] + list(pillar_score_map.keys())
    ].melt(
        id_vars=["REF_AREA_LABEL", "economy_label"],
        value_vars=list(pillar_score_map.keys()),
        var_name="score_type",
        value_name="score"
    )

    pillar_plot_df["pillar"] = pillar_plot_df["score_type"].map(pillar_score_map)
    pillar_plot_df["pillar"] = pd.Categorical(
        pillar_plot_df["pillar"],
        categories=PILLAR_ORDER,
        ordered=True
    )

    fig = px.bar(
        pillar_plot_df,
        x="economy_label",
        y="score",
        color="pillar",
        barmode="group",
        text="score",
        title="Three-Dimension Profile: Education, Participation, and Infrastructure",
        labels={
            "economy_label": "Economy",
            "score": "Score",
            "pillar": "Dimension"
        },
        color_discrete_map=PILLAR_COLOR_MAP,
        category_orders={
            "economy_label": COUNTRY_LABEL_ORDER,
            "pillar": PILLAR_ORDER
        }
    )

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 105],
        legend_title_text="Dimension"
    )

    return fig


def plot_three_dimension_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="education_equity_score",
        y="formal_financial_participation_score",
        color="REF_AREA_LABEL",
        text="economy_label",
        size="financial_access_infrastructure_score",
        size_max=55,
        title="Three-Dimension View: Education Equity, Financial Participation, and Access Infrastructure",
        labels={
            "education_equity_score": "Mathematics Learning Equity Score",
            "formal_financial_participation_score": "Formal Financial Participation Score",
            "financial_access_infrastructure_score": "Financial Access Infrastructure Score",
            "REF_AREA_LABEL": "Economy",
            "economy_label": "Economy"
        },
        color_discrete_map=COUNTRY_COLOR_MAP,
        category_orders={
            "REF_AREA_LABEL": COUNTRY_ORDER,
            "economy_label": COUNTRY_LABEL_ORDER
        },
        hover_name="REF_AREA_LABEL",
        hover_data={
            "economy_label": False,
            "education_equity_score": ":.1f",
            "formal_financial_participation_score": ":.1f",
            "financial_access_infrastructure_score": ":.1f"
        }
    )

    # Avoid text being clipped by large bubbles.
    fig.update_traces(cliponaxis=False)

    for trace in fig.data:
        if trace.name == "Korea, Rep.":
            trace.textposition = "bottom center"
        elif trace.name == "Hong Kong SAR, China":
            trace.textposition = "top center"
        elif trace.name == "Thailand":
            trace.textposition = "top left"
        elif trace.name == "Indonesia":
            trace.textposition = "top center"

    fig.update_layout(
        template="plotly_white",
        title={
            "text": (
                "Three-Dimension View: Education Equity, Financial Participation, "
                "and Access Infrastructure<br>"
                "<sup>Bubble size represents the financial access infrastructure score.</sup>"
            ),
            "x": 0.02,
            "xanchor": "left"
        },
        legend_title_text="Economy",
        margin=dict(t=110, r=40, b=60, l=60),
        yaxis=dict(range=[20, 95]),
        xaxis=dict(range=[63, 96])
    )

    return fig


def plot_indicator_bar(df: pd.DataFrame, indicator_col: str, title: str, y_label: str):
    plot_df = df.copy()

    fig = px.bar(
        plot_df,
        x="economy_label",
        y=indicator_col,
        color="REF_AREA_LABEL",
        text=indicator_col,
        title=title,
        labels={
            "economy_label": "Economy",
            indicator_col: y_label,
            "REF_AREA_LABEL": "Economy"
        },
        color_discrete_map=COUNTRY_COLOR_MAP,
        category_orders={
            "economy_label": COUNTRY_LABEL_ORDER,
            "REF_AREA_LABEL": COUNTRY_ORDER
        },
        hover_name="REF_AREA_LABEL"
    )

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        showlegend=False
    )

    return fig


# ============================================================
# Load data
# ============================================================

df = load_analysis_data()
forecast_df = load_forecast_data()


# ============================================================
# Header
# ============================================================

st.title("Project 3: SDG 4 & 8 - Mathematics Learning Equity and Financial Inclusion Ecosystem Explorer")

st.markdown(
    """
The project looks at financial inclusion as an ecosystem with three connected dimensions:

1. **Mathematics-related learning equity** — the education-side foundation.  
2. **Formal financial participation** — the people-side usage of formal financial tools.  
3. **Financial access infrastructure** — the environment-side access and service context.  

It compares four selected Asian economies: **Korea, Hong Kong, Thailand, and Indonesia**.
"""
)


# ============================================================
# Tabs
# ============================================================

tab_country, tab_charts, tab_indicators, tab_forecast, tab_notes = st.tabs(
    [
        "Country Profile",
        "Ecosystem Charts",
        "Indicator Details",
        "Predictive Outlook",
        "Data Notes"
    ]
)

# ============================================================
# Tab 1: Country Profile
# ============================================================

with tab_country:
    st.subheader("Country Profile")

    selected_country = st.selectbox(
        "Select an economy",
        options=COUNTRY_ORDER,
        format_func=lambda x: COUNTRY_LABEL_MAP.get(x, x),
        key="country_profile_selector"
    )

    selected_country_row = df[df["REF_AREA_LABEL"] == selected_country].iloc[0]

    # Centered ecosystem score with a short explanation
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 0.8rem; margin-bottom: 1.6rem;">
            <div style="font-size: 1.4rem; font-weight: 700; color: #606673; margin-bottom: 0.2rem;">
                Ecosystem score
            </div>
            <div style="font-size: 2.2rem; font-weight: 500; line-height: 1; margin-bottom: 0.7rem;">
                {format_score(selected_country_row.get("financial_inclusion_ecosystem_score"))}
            </div>
            <div style="max-width: 680px; margin: 0 auto; font-size: 0.95rem; color: #6b7280; line-height: 1.55;">
                This exploratory composite score combines mathematics-learning equity,
                formal financial participation, and financial access infrastructure.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Three-pillar profile")

    selected_profile = pd.DataFrame(
        {
            "Dimension": [
                "Mathematics learning equity",
                "Formal financial participation",
                "Financial access infrastructure"
            ],
            "Score": [
                selected_country_row.get("education_equity_score"),
                selected_country_row.get("formal_financial_participation_score"),
                selected_country_row.get("financial_access_infrastructure_score")
            ]
        }
    )

    fig_country_profile = px.bar(
        selected_profile,
        x="Dimension",
        y="Score",
        color="Dimension",
        text="Score",
        labels={
            "Dimension": "Dimension",
            "Score": "Score"
        },
        color_discrete_map=PILLAR_COLOR_MAP,
        category_orders={"Dimension": PILLAR_ORDER}
    )

    fig_country_profile.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )

    fig_country_profile.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis_range=[0, 105],
        title_text="",   # prevents the "undefined" title from appearing
        margin=dict(t=20, r=40, b=60, l=60)
    )

    st.plotly_chart(
        fig_country_profile,
        use_container_width=True,
        key=f"country_profile_pillar_chart_{selected_country}"
    )

    st.markdown("### Raw data for selected economy")

    display_cols = [
        col for col in [
            "REF_AREA_LABEL",
            "edu_year",
            "findex_year",
            "fas_year",
            "education_equity_score",
            "formal_financial_participation_score",
            "financial_access_infrastructure_score",
            "financial_inclusion_ecosystem_score"
        ] + EDUCATION_RAW_COLS + FINDEX_RAW_COLS + FAS_RAW_COLS
        if col in df.columns
    ]

    st.dataframe(
        selected_country_row[display_cols].to_frame(name="Value"),
        use_container_width=True
    )


# ============================================================
# Tab 2: Ecosystem Charts
# ============================================================

with tab_charts:
    st.subheader("8-1. Financial Inclusion Ecosystem Score")

    st.markdown(
        """
This score gives a high-level view of the three-part framework. It combines mathematics-related
learning equity, formal financial participation, and financial access infrastructure. The purpose is not
to rank economies as definitively better or worse, but to summarize how strongly each economy appears
across the selected indicators.
"""
    )

    st.plotly_chart(
        plot_ecosystem_score(df),
        use_container_width=True,
        key="charts_ecosystem_score_chart"
    )

    st.divider()

    st.subheader("8-2. Three-Dimension Profile")

    st.markdown(
        """
This chart separates the ecosystem score into its three components. It helps show whether an economy’s
position is driven more by education equity, people-side financial participation, or financial access infrastructure.
"""
    )

    st.plotly_chart(
        plot_three_dimension_profile(df),
        use_container_width=True,
        key="charts_three_dimension_profile"
    )

    st.divider()

    st.subheader("8-3. Three-Dimension Relationship View")

    st.markdown(
        """
This is the central cross-SDG comparison. It connects the education-side foundation with the people-side
financial inclusion outcome. Bubble size represents financial access infrastructure, adding the environment-side
context to the same view.

The chart should be read as exploratory. With only four economies and near-year data, it can suggest patterns
but cannot prove causal conclusions.
"""
    )

    st.plotly_chart(
        plot_three_dimension_scatter(df),
        use_container_width=True,
        key="charts_three_dimension_scatter"
    )


# ============================================================
# Tab 3: Indicator Details
# ============================================================

with tab_indicators:
    st.subheader("Education, Financial Participation, and Infrastructure Indicators")

    st.markdown(
        """
This section lets users zoom into the indicators behind the three scores.
Education indicators are parity indexes, so values closer to 1 indicate more equal outcomes across groups.
Findex indicators are percentage-based usage measures. FAS indicators describe financial access infrastructure
and use different units.
"""
    )

    indicator_group = st.radio(
        "Select an indicator group",
        [
            "Education equity indicators",
            "Formal financial participation indicators",
            "Financial access infrastructure indicators"
        ],
        horizontal=True,
        key="indicator_group_selector"
    )

    if indicator_group == "Education equity indicators":
        available_cols = [col for col in EDUCATION_RAW_COLS if col in df.columns]

        selected_indicator = st.selectbox(
            "Select an education indicator",
            available_cols,
            format_func=pretty_name,
            key="education_indicator_selector"
        )

        st.info(
            "Education indicators are parity indexes. A value of 1 means the compared groups have equal "
            "mathematics-related outcomes. Values farther from 1 indicate larger inequality in either direction."
        )

        fig_indicator = plot_indicator_bar(
            df,
            selected_indicator,
            title=pretty_name(selected_indicator),
            y_label="Parity index"
        )

        fig_indicator.add_hline(
            y=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Parity = 1",
            annotation_position="top left"
        )

        st.plotly_chart(
            fig_indicator,
            use_container_width=True,
            key=f"indicator_chart_education_{selected_indicator}"
        )

    elif indicator_group == "Formal financial participation indicators":
        available_cols = [col for col in FINDEX_RAW_COLS if col in df.columns]

        selected_indicator = st.selectbox(
            "Select a Findex participation indicator",
            available_cols,
            format_func=pretty_name,
            key="findex_indicator_selector"
        )

        st.info(
            "Findex indicators are interpreted as people-side financial participation measures. "
            "They show whether adults are actually using formal financial tools."
        )

        fig_indicator = plot_indicator_bar(
            df,
            selected_indicator,
            title=pretty_name(selected_indicator),
            y_label="Percentage of adults"
        )

        fig_indicator.update_layout(yaxis_range=[0, 105])

        st.plotly_chart(
            fig_indicator,
            use_container_width=True,
            key=f"indicator_chart_findex_{selected_indicator}"
        )

    else:
        available_cols = [col for col in FAS_RAW_COLS if col in df.columns]

        selected_indicator = st.selectbox(
            "Select a FAS infrastructure indicator",
            available_cols,
            format_func=pretty_name,
            key="fas_indicator_selector"
        )

        st.info(
            "FAS indicators describe the environment-side financial access context. "
            "They are not direct measures of people’s behavior, but they show whether formal financial services "
            "and access points are available."
        )

        fig_indicator = plot_indicator_bar(
            df,
            selected_indicator,
            title=pretty_name(selected_indicator),
            y_label=pretty_name(selected_indicator)
        )

        st.plotly_chart(
            fig_indicator,
            use_container_width=True,
            key=f"indicator_chart_fas_{selected_indicator}"
        )

    st.markdown("### Indicator data")

    detail_cols = [
        "REF_AREA_LABEL",
        "economy_label"
    ]

    if indicator_group == "Education equity indicators":
        detail_cols += [col for col in EDUCATION_RAW_COLS if col in df.columns]
    elif indicator_group == "Formal financial participation indicators":
        detail_cols += [col for col in FINDEX_RAW_COLS if col in df.columns]
    else:
        detail_cols += [col for col in FAS_RAW_COLS if col in df.columns]

    st.dataframe(df[detail_cols], use_container_width=True)


# ============================================================
# Tab 4: Predictive Outlook
# ============================================================

with tab_forecast:
    st.subheader("Predictive Outlook: Financial Access Infrastructure")

    st.markdown(
        """
The predictive outlook uses historical IMF Financial Access Survey data because infrastructure indicators
have more consistent annual observations than Findex survey indicators.

This is not a forecast of people’s financial behavior. It is an exploratory outlook for the financial access
environment: how selected access points or financial infrastructure measures may continue to change if recent
historical trends continue.
"""
    )

    if forecast_df.empty:
        st.warning(
            "fas_forecast_data.csv was not found. Please add it to the same folder as app.py "
            "to enable the predictive outlook section."
        )
    else:
        available_forecast_countries = [
            country for country in COUNTRY_ORDER
            if country in forecast_df["REF_AREA_LABEL"].unique()
        ]

        forecast_country = st.selectbox(
            "Select economy for prediction",
            options=available_forecast_countries,
            format_func=lambda x: COUNTRY_LABEL_MAP.get(x, x),
            key="forecast_country_selector"
        )

        country_forecast_df = forecast_df[
            forecast_df["REF_AREA_LABEL"] == forecast_country
        ].copy()

        indicator_options = (
            country_forecast_df[["short_indicator", "indicator_label"]]
            .drop_duplicates()
            .sort_values("indicator_label")
        )

        indicator_label_to_short = dict(
            zip(indicator_options["indicator_label"], indicator_options["short_indicator"])
        )

        selected_indicator_label = st.selectbox(
            "Select infrastructure indicator",
            options=list(indicator_label_to_short.keys()),
            key="forecast_indicator_selector"
        )

        selected_indicator = indicator_label_to_short[selected_indicator_label]

        selected_hist = country_forecast_df[
            country_forecast_df["short_indicator"] == selected_indicator
        ].copy()

        selected_hist = selected_hist.sort_values("year").reset_index(drop=True)

        if selected_hist.shape[0] < 2:
            st.warning("Not enough historical data points to create a projection.")
        else:
            x = selected_hist["year"].astype(float).values
            y = selected_hist["value"].astype(float).values

            slope, intercept = np.polyfit(x, y, 1)

            selected_hist["series"] = "Historical"

            last_year = int(selected_hist["year"].max())
            last_value = float(selected_hist["value"].iloc[-1])

            future_years = list(range(last_year + 1, last_year + 6))
            future_values = [
                max(0, last_value + slope * (year - last_year))
                for year in future_years
            ]

            future_df = pd.DataFrame(
                {
                    "REF_AREA_LABEL": forecast_country,
                    "short_indicator": selected_indicator,
                    "indicator_label": selected_indicator_label,
                    "year": future_years,
                    "value": future_values,
                    "series": "Projected"
                }
            )

            last_actual = selected_hist.tail(1).copy()
            last_actual["series"] = "Projected"

            plot_df = pd.concat(
                [
                    selected_hist[["year", "value", "series"]],
                    last_actual[["year", "value", "series"]],
                    future_df[["year", "value", "series"]]
                ],
                ignore_index=True
            )

            fig_forecast = px.line(
                plot_df,
                x="year",
                y="value",
                color="series",
                line_dash="series",
                markers=True,
                title=f"{selected_indicator_label}: Historical Trend and 5-Year Projection",
                labels={
                    "year": "Year",
                    "value": selected_indicator_label,
                    "series": "Series"
                },
                color_discrete_map={
                    "Historical": "#0B5CAD",
                    "Projected": "#7CC7FF"
                },
                line_dash_map={
                    "Historical": "solid",
                    "Projected": "dash"
                },
                category_orders={
                    "series": ["Historical", "Projected"]
                }
            )

            fig_forecast.update_traces(marker=dict(size=7), line=dict(width=3))
            fig_forecast.update_layout(template="plotly_white", legend_title_text="Series")

            st.plotly_chart(
                fig_forecast,
                use_container_width=True,
                key=f"forecast_chart_{forecast_country}_{selected_indicator}"
            )

            st.markdown(
                f"""
**Interpretation:**  
This projection uses a simple country-specific linear trend based on historical FAS data for
**{COUNTRY_LABEL_MAP.get(forecast_country, forecast_country)}**. It should be interpreted as a directional scenario,
not a precise forecast. The goal is to show how the financial access environment may continue to change if recent
historical patterns continue.
"""
            )

            projection_summary = pd.concat(
                [
                    selected_hist[["year", "value"]].assign(type="Historical"),
                    future_df[["year", "value"]].assign(type="Projected")
                ],
                ignore_index=True
            )

            st.dataframe(projection_summary, use_container_width=True)


# ============================================================
# Tab 5: Data Notes
# ============================================================

with tab_notes:
    st.subheader("Data Sources and Methodology Notes")

    st.markdown(
        """
### Data sources

- **World Bank Education Statistics**: mathematics-related learning equity indicators  
- **World Bank Global Findex**: formal financial participation indicators  
- **IMF Financial Access Survey**: financial access infrastructure indicators  

### Year alignment

Education and FAS indicators use **2018** values where available.  
Findex indicators use **2017**, the nearest available survey year before 2018.  
Because the datasets do not all report every year, the final comparison should be interpreted as a near-year exploratory comparison, not as a perfectly same-year causal analysis.

### Score interpretation

- **Mathematics learning equity score**: based on parity indicators. Values closer to 1 are treated as more equitable.  
- **Formal financial participation score**: average of selected Findex usage percentages.  
- **Financial access infrastructure score**: relative min-max score across selected economies because FAS indicators use different units.  
- **Financial inclusion ecosystem score**: exploratory composite score combining the three pillars.  

### Limitations

This project uses mathematics-related learning equity as an indirect education-side measure because direct adult numeracy data was limited.
The analysis includes only four economies and combines near-year data from multiple sources.
The composite scores depend on the selected indicators, scoring method, and available data coverage.
Therefore, the findings should be interpreted as descriptive patterns rather than statistical or causal conclusions.
"""
    )

    with st.expander("Optional diagnostic check: score correlations"):
        st.markdown(
            """
This optional diagnostic check looks at whether the three pillar scores move in a similar direction across
the four selected economies. The correlations are positive, but because the sample includes only four economies,
they should not be interpreted as statistical evidence.
"""
        )

        score_cols = [
            "education_equity_score",
            "formal_financial_participation_score",
            "financial_access_infrastructure_score"
        ]

        available_score_cols = [col for col in score_cols if col in df.columns]

        if len(available_score_cols) >= 2:
            st.dataframe(df[available_score_cols].corr(), use_container_width=True)
        else:
            st.info("Not enough score columns available to compute correlations.")

    with st.expander("Final analysis data"):
        st.dataframe(df, use_container_width=True)
