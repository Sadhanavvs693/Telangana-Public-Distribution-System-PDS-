import streamlit as st
import pandas as pd
import numpy as np
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telangana PDS Analytics Dashboard",
    page_icon="🍚",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data_processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

@st.cache_data
def load_csv(file_path):

    if os.path.exists(file_path):

        try:
            return pd.read_csv(file_path)

        except Exception as e:

            st.warning(
                f"Could not read {os.path.basename(file_path)}"
            )

            return pd.DataFrame()

    return pd.DataFrame()


# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = load_csv(
    os.path.join(
        DATA_DIR,
        "pds_master_dataset.csv"
    )
)


# ============================================================
# LOAD OUTPUT FILES
# ============================================================

transaction_outliers = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "transaction_outliers.csv"
    )
)

ratio_outliers = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "ration_card_ratio_outliers.csv"
    )
)

rice_outliers = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "rice_transaction_outliers.csv"
    )
)

isolation_anomalies = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "isolation_forest_shop_anomalies.csv"
    )
)

high_priority = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "high_priority_anomalous_shops.csv"
    )
)

rf_predictions = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_future_random_forest_predictions.csv"
    )
)

rf_metrics = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_future_random_forest_metrics.csv"
    )
)

xgb_predictions = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_future_xgboost_predictions.csv"
    )
)

xgb_metrics = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_future_xgboost_metrics.csv"
    )
)

xgb_importance = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_future_xgboost_feature_importance.csv"
    )
)
# ============================================================
# LOAD CLUSTERING OUTPUT FILES
# ============================================================

shop_clusters = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "shop_clustering_results.csv"
    )
)

kmeans_profiles = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_cluster_profiles.csv"
    )
)

dbscan_distribution = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "dbscan_cluster_distribution.csv"
    )
)

district_clusters = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "district_cluster_distribution.csv"
    )
)

pca_variance = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "pca_explained_variance.csv"
    )
)

# ============================================================
# CHECK MASTER DATA
# ============================================================

if df.empty:

    st.error(
        "pds_master_dataset.csv could not be loaded."
    )

    st.stop()


# ============================================================
# CLEAN DATA TYPES
# ============================================================

numeric_columns = [
    "year",
    "month",
    "noOfRcs",
    "noOfTrans",
    "totalRcs",
    "totalUnits",
    "riceAfsc",
    "riceFsc",
    "riceAap",
    "wheat",
    "sugar",
    "rgdal",
    "kerosene",
    "totalAmount",
    "salt",
    "latitude",
    "longitude"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("🍚 Telangana PDS")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 District Analysis",
        "📈 Transaction Analysis",
        "🚨 Anomaly Detection",
        "🤖 ML Prediction",
        "🔍 Shop Investigation",
        "💡 Business Insights",
        "🧩 Shop Clustering"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🍚 Telangana PDS Data Analytics Dashboard"
)

st.caption(
    "District Analysis • Transaction Analysis • "
    "Anomaly Detection • Machine Learning"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("🏠 Project Overview")

    total_records = len(df)

    total_shops = (
        df["shopNo"].nunique()
        if "shopNo" in df.columns
        else 0
    )

    total_districts = (
        df["distName"].nunique()
        if "distName" in df.columns
        else 0
    )

    total_transactions = (
        df["noOfTrans"].sum()
        if "noOfTrans" in df.columns
        else 0
    )

    total_ration_cards = (
        df["totalRcs"].sum()
        if "totalRcs" in df.columns
        else 0
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Records",
        f"{total_records:,}"
    )

    c2.metric(
        "Shops",
        f"{total_shops:,}"
    )

    c3.metric(
        "Districts",
        f"{total_districts:,}"
    )

    c4.metric(
        "Transactions",
        f"{total_transactions:,.0f}"
    )

    c5.metric(
        "Ration Cards",
        f"{total_ration_cards:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("Dataset Information")

    overview = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Unique Shops",
            "Unique Districts",
            "Years",
            "Transaction Records"
        ],

        "Value": [
            len(df),
            len(df.columns),
            df["shopNo"].nunique(),
            df["distName"].nunique(),
            df["year"].nunique(),
            df["noOfTrans"].notna().sum()
        ]
    })

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Available Years")

    years = sorted(
        df["year"].dropna().unique()
    )

    st.write(
        ", ".join(
            str(int(year))
            for year in years
        )
    )


# ============================================================
# DISTRICT ANALYSIS
# ============================================================

elif page == "📊 District Analysis":

    st.header("📊 District Analysis")

    # --------------------------------------------------------
    # DISTRICT SUMMARY
    # --------------------------------------------------------

    district_summary = (
        df.groupby(
            "distName",
            dropna=False
        )
        .agg(
            shops=(
                "shopNo",
                "nunique"
            ),

            transactions=(
                "noOfTrans",
                "sum"
            ),

            avg_transactions=(
                "noOfTrans",
                "mean"
            ),

            ration_cards=(
                "totalRcs",
                "sum"
            )
        )
        .reset_index()
    )

    district_summary = district_summary.rename(
        columns={
            "distName": "District"
        }
    )

    district_summary = district_summary.sort_values(
        "transactions",
        ascending=False
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    top_district = (
        district_summary.iloc[0]["District"]
        if not district_summary.empty
        else "N/A"
    )

    top_transactions = (
        district_summary.iloc[0]["transactions"]
        if not district_summary.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Districts",
        f"{len(district_summary):,}"
    )

    c2.metric(
        "Shops",
        f"{df['shopNo'].nunique():,}"
    )

    c3.metric(
        "Top District",
        str(top_district)
    )

    c4.metric(
        "Top District Transactions",
        f"{top_transactions:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # DISTRICT TRANSACTION CHART
    # --------------------------------------------------------

    st.subheader(
        "📊 Total Transactions by District"
    )

    district_chart = (
        district_summary[
            [
                "District",
                "transactions"
            ]
        ]
        .set_index("District")
    )

    st.bar_chart(
        district_chart
    )

    # --------------------------------------------------------
    # DISTRICT SHOP COUNT
    # --------------------------------------------------------

    st.subheader(
        "🏪 Shops by District"
    )

    shop_chart = (
        district_summary[
            [
                "District",
                "shops"
            ]
        ]
        .set_index("District")
        .sort_values(
            "shops",
            ascending=False
        )
    )

    st.bar_chart(
        shop_chart
    )

    # --------------------------------------------------------
    # DISTRICT TABLE
    # --------------------------------------------------------

    st.subheader(
        "District Summary Table"
    )

    display_district = district_summary.copy()

    display_district[
        "transactions"
    ] = display_district[
        "transactions"
    ].round(0)

    display_district[
        "avg_transactions"
    ] = display_district[
        "avg_transactions"
    ].round(2)

    display_district[
        "ration_cards"
    ] = display_district[
        "ration_cards"
    ].round(0)

    st.dataframe(
        display_district,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

elif page == "📈 Transaction Analysis":

    st.header("📈 Transaction Analysis")

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_transactions = (
        df["noOfTrans"].sum()
    )

    average_transactions = (
        df["noOfTrans"].mean()
    )

    maximum_transactions = (
        df["noOfTrans"].max()
    )

    transaction_records = (
        df["noOfTrans"].notna().sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,.0f}"
    )

    c2.metric(
        "Average / Record",
        f"{average_transactions:,.2f}"
    )

    c3.metric(
        "Maximum / Record",
        f"{maximum_transactions:,.0f}"
    )

    c4.metric(
        "Transaction Records",
        f"{transaction_records:,}"
    )

    st.divider()

    # --------------------------------------------------------
    # MONTHLY TRANSACTION TREND
    # --------------------------------------------------------

    st.subheader(
        "📅 Monthly Transaction Trend"
    )

    monthly = (
        df.groupby(
            [
                "year",
                "month"
            ],
            as_index=False
        )
        .agg(
            transactions=(
                "noOfTrans",
                "sum"
            )
        )
    )

    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(int).astype(str)
        + "-"
        + monthly["month"].astype(int).astype(str)
        + "-01",
        errors="coerce"
    )

    monthly = monthly.dropna(
        subset=["date"]
    ).sort_values(
        "date"
    )

    monthly_chart = (
        monthly[
            [
                "date",
                "transactions"
            ]
        ]
        .set_index("date")
    )

    st.line_chart(
        monthly_chart
    )

    # --------------------------------------------------------
    # YEARLY TRANSACTION SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📅 Yearly Transaction Summary"
    )

    yearly = (
        df.groupby(
            "year",
            as_index=False
        )
        .agg(
            total_transactions=(
                "noOfTrans",
                "sum"
            ),

            average_transactions=(
                "noOfTrans",
                "mean"
            ),

            transaction_records=(
                "noOfTrans",
                "count"
            )
        )
        .sort_values(
            "year"
        )
    )

    yearly[
        "total_transactions"
    ] = yearly[
        "total_transactions"
    ].round(0)

    yearly[
        "average_transactions"
    ] = yearly[
        "average_transactions"
    ].round(2)

    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )

    yearly_chart = (
        yearly[
            [
                "year",
                "total_transactions"
            ]
        ]
        .set_index("year")
    )

    st.line_chart(
        yearly_chart
    )

    # --------------------------------------------------------
    # TOP 10 SHOPS
    # --------------------------------------------------------

    st.subheader(
        "🏆 Top 10 Shops by Transactions"
    )

    top_shops = (
        df.groupby(
            [
                "shopNo",
                "distName"
            ],
            as_index=False
        )
        .agg(
            total_transactions=(
                "noOfTrans",
                "sum"
            )
        )
        .sort_values(
            "total_transactions",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_shops,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🚨 Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    st.info(
        "Anomalies are statistical signals requiring "
        "further investigation. They are not confirmed fraud."
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transaction Outliers",
        f"{len(transaction_outliers):,}"
    )

    c2.metric(
        "Ratio Outliers",
        f"{len(ratio_outliers):,}"
    )

    c3.metric(
        "Rice Outliers",
        f"{len(rice_outliers):,}"
    )

    c4.metric(
        "Isolation Forest",
        f"{len(isolation_anomalies):,}"
    )

    st.divider()

    # --------------------------------------------------------
    # HIGH PRIORITY
    # --------------------------------------------------------

    st.subheader(
        "🔴 High-Priority Investigation Shops"
    )

    if not high_priority.empty:

        st.metric(
            "High-Priority Shops",
            f"{len(high_priority):,}"
        )

        st.dataframe(
            high_priority.head(100),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "High-priority anomaly file not found."
        )

    # --------------------------------------------------------
    # TRANSACTION OUTLIERS
    # --------------------------------------------------------

    st.subheader(
        "Transaction Outliers"
    )

    if not transaction_outliers.empty:

        st.dataframe(
            transaction_outliers.head(100),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.header(
        "🤖 Future Transaction Prediction"
    )

    st.info(
        "The ML models predict whether a shop is likely "
        "to have high transaction activity in the following month."
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    model_comparison = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy (%)": [
            96.52,
            96.79,
            96.76
        ],

        "Precision (%)": [
            95.06,
            95.01,
            95.26
        ],

        "Recall (%)": [
            91.53,
            92.66,
            92.28
        ],

        "F1 Score (%)": [
            93.26,
            93.82,
            93.75
        ],

        "ROC-AUC (%)": [
            99.17,
            99.40,
            99.40
        ]
    })

    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Selected Final Model: Random Forest"
    )

    st.write(
        "Random Forest achieved the highest overall "
        "accuracy, recall and F1-score."
    )

    # --------------------------------------------------------
    # XGBOOST FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🔎 XGBoost Feature Importance"
    )

    if not xgb_importance.empty:

        st.dataframe(
            xgb_importance.head(15),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "XGBoost feature importance file not found."
        )

    # --------------------------------------------------------
    # RANDOM FOREST PREDICTIONS
    # --------------------------------------------------------

    st.subheader(
        "Random Forest Predictions"
    )

    if not rf_predictions.empty:

        st.dataframe(
            rf_predictions.head(100),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Random Forest prediction file not found."
        )


# ============================================================
# SHOP INVESTIGATION
# ============================================================

elif page == "🔍 Shop Investigation":

    st.header(
        "🔍 Shop Investigation"
    )

    # --------------------------------------------------------
    # SHOP SELECTION
    # --------------------------------------------------------

    shops = sorted(
        df["shopNo"]
        .dropna()
        .unique()
    )

    selected_shop = st.selectbox(
        "Select Shop",
        shops
    )

    history = df[
        df["shopNo"] == selected_shop
    ].copy()

    if history.empty:

        st.warning(
            "No data available for this shop."
        )

        st.stop()

    # --------------------------------------------------------
    # SHOP DETAILS
    # --------------------------------------------------------

    district = (
        history["distName"].dropna().iloc[0]
        if history["distName"].notna().any()
        else "N/A"
    )

    total_trans = (
        history["noOfTrans"].sum()
    )

    avg_trans = (
        history["noOfTrans"].mean()
    )

    max_trans = (
        history["noOfTrans"].max()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Shop",
        str(selected_shop)
    )

    c2.metric(
        "District",
        str(district)
    )

    c3.metric(
        "Total Transactions",
        f"{total_trans:,.0f}"
    )

    c4.metric(
        "Average Transactions",
        f"{avg_trans:,.2f}"
    )

    st.divider()

    # --------------------------------------------------------
    # HISTORICAL TRANSACTIONS
    # --------------------------------------------------------

    st.subheader(
        "📈 Historical Transactions"
    )

    history["date"] = pd.to_datetime(
        history["year"].astype(int).astype(str)
        + "-"
        + history["month"].astype(int).astype(str)
        + "-01",
        errors="coerce"
    )

    history = history.dropna(
        subset=[
            "date",
            "noOfTrans"
        ]
    ).sort_values(
        "date"
    )

    history_chart = (
        history[
            [
                "date",
                "noOfTrans"
            ]
        ]
        .set_index("date")
    )

    st.line_chart(
        history_chart
    )

    # --------------------------------------------------------
    # SHOP HISTORY TABLE
    # --------------------------------------------------------

    st.subheader(
        "Shop Transaction History"
    )

    history_table = history[
        [
            "date",
            "year",
            "month",
            "noOfRcs",
            "noOfTrans",
            "totalRcs",
            "totalUnits"
        ]
    ].copy()

    st.dataframe(
        history_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "💡 Business Insights":

    st.header(
        "💡 Business Insights"
    )

    st.subheader(
        "Key Findings"
    )

    insights = [

        "Hyderabad has the highest transaction activity among districts.",

        "Medchal also shows substantially high transaction activity.",

        "Historical transaction behaviour is the strongest predictor "
        "of future transaction activity.",

        "Previous-month transaction behaviour is highly important "
        "for predicting the following month.",

        "Recent 2–3 month transaction history improves model prediction.",

        "Multiple anomaly detection methods identify shops "
        "requiring further investigation.",

        "The strongest agreement between anomaly methods occurred "
        "in Hyderabad.",

        "Anomaly detection identifies statistical signals and "
        "does not confirm fraud.",

        "Random Forest was selected as the final ML model.",

        "Random Forest achieved 96.79% accuracy and 93.82% F1-score."
    ]

    for number, insight in enumerate(
        insights,
        1
    ):

        st.write(
            f"**{number}.** {insight}"
        )

    st.divider()

    st.subheader(
        "Recommended Actions"
    )

    actions = [

        "Investigate shops repeatedly flagged by multiple "
        "anomaly detection methods.",

        "Review unusually high transaction volumes.",

        "Monitor persistent month-to-month high transaction activity.",

        "Use historical transaction patterns for proactive monitoring.",

        "Validate statistical anomalies with operational information "
        "before taking action."
    ]

    for action in actions:

        st.write(
            f"• {action}"
        )


# ============================================================
# SHOP CLUSTERING
# ============================================================

elif page == "🧩 Shop Clustering":

    st.header("🧩 Shop Clustering Analysis")

    st.write(
        "Explore shop activity patterns using "
        "PCA, K-Means clustering and DBSCAN."
    )

    # --------------------------------------------------------
    # CLUSTERING OVERVIEW
    # --------------------------------------------------------

    st.subheader("Clustering Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Shops Analysed",
            f"{len(shop_clusters):,}"
        )

    with col2:
        if "kmeans_cluster" in shop_clusters.columns:
            st.metric(
                "K-Means Clusters",
                shop_clusters["kmeans_cluster"].nunique()
            )

    with col3:
        if "dbscan_cluster" in shop_clusters.columns:
            noise_count = (
                shop_clusters["dbscan_cluster"] == -1
            ).sum()

            st.metric(
                "DBSCAN Noise Shops",
                f"{noise_count:,}"
            )

    st.divider()

    # --------------------------------------------------------
    # K-MEANS CLUSTER PROFILES
    # --------------------------------------------------------

    st.subheader("K-Means Cluster Profiles")

    st.dataframe(
        kmeans_profiles,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DBSCAN CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("DBSCAN Cluster Distribution")

    st.dataframe(
        dbscan_distribution,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DISTRICT CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("District-wise Cluster Distribution")

    st.dataframe(
        district_clusters,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PCA EXPLAINED VARIANCE
    # --------------------------------------------------------

    st.subheader("PCA Explained Variance")

    st.dataframe(
        pca_variance,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CLUSTERING VISUALIZATIONS
    # --------------------------------------------------------

    st.subheader("Clustering Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            os.path.join(
                OUTPUT_DIR,
                "kmeans_pca_clusters.png"
            ),
            caption="K-Means Clusters using PCA",
            use_container_width=True
        )

    with col2:
        st.image(
            os.path.join(
                OUTPUT_DIR,
                "kmeans_cluster_distribution.png"
            ),
            caption="K-Means Cluster Distribution",
            use_container_width=True
        )

    st.image(
        os.path.join(
            OUTPUT_DIR,
            "dbscan_cluster_distribution.png"
        ),
        caption="DBSCAN Cluster Distribution",
        use_container_width=True
    )

    st.info(
        "DBSCAN noise points indicate shops that do not meet "
        "the selected density criteria. They should not be "
        "interpreted as poor-performing or fraudulent shops."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Telangana PDS Analytics Project | "
    "Statistical Analysis + Anomaly Detection + Machine Learning"
)