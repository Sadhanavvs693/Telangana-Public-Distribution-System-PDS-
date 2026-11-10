import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

# ==================================================
# 1. LOAD SHOP SUMMARY
# ==================================================

shop_summary = pd.read_csv(
    OUTPUT_DIR / "shop_summary.csv"
)

print("=" * 60)
print("TELANGANA PDS - PCA AND CLUSTERING ANALYSIS")
print("=" * 60)

print("\nOriginal shop summary shape:", shop_summary.shape)

# ==================================================
# 2. SEPARATE SHOPS WITH TRANSACTION DATA
# ==================================================

shops_with_transactions = shop_summary[
    shop_summary["transaction_months"] > 0
].copy()

shops_without_transactions = shop_summary[
    shop_summary["transaction_months"] == 0
].copy()

print("\nShops with transaction records:",
      len(shops_with_transactions))

print("Shops without transaction records:",
      len(shops_without_transactions))

# ==================================================
# 3. SELECT CLUSTERING FEATURES
# ==================================================

FEATURES = [
    "avg_monthly_transactions",
    "avg_monthly_ration_cards",
    "transactions_per_ration_card",
    "total_rice",
    "total_sugar"
]

X = shops_with_transactions[FEATURES].copy()

print("\nSelected clustering features:")
print(FEATURES)

print("\nFeature dataset shape:", X.shape)

# ==================================================
# 4. CHECK MISSING AND INFINITE VALUES
# ==================================================

X = X.replace([np.inf, -np.inf], np.nan)

print("\nMissing values before treatment:")
print(X.isna().sum())

# Remove rows with missing clustering features
valid_mask = X.notna().all(axis=1)

X = X.loc[valid_mask].copy()

shops_for_clustering = (
    shops_with_transactions.loc[valid_mask].copy()
)

print("\nValid shops for clustering:", len(X))
print("Excluded shops due to missing features:",
      len(shops_with_transactions) - len(X))

# ==================================================
# 5. LOG TRANSFORMATION
# ==================================================

LOG_FEATURES = [
    "avg_monthly_transactions",
    "avg_monthly_ration_cards",
    "total_rice",
    "total_sugar"
]

for column in LOG_FEATURES:
    X[column] = np.log1p(X[column])

print("\nLog transformation completed.")

print("\nSkewness after log transformation:")
print(X[LOG_FEATURES].skew().round(2))

# ==================================================
# 6. FEATURE STANDARDIZATION
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature standardization completed.")

print("\nScaled dataset shape:", X_scaled.shape)

print("\nScaled feature means:")
print(np.round(X_scaled.mean(axis=0), 4))

print("\nScaled feature standard deviations:")
print(np.round(X_scaled.std(axis=0), 4))

# ==================================================
# 7. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ==================================================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

pca_summary = pd.DataFrame({
    "Principal Component": [
        f"PC{i+1}" for i in range(len(explained_variance))
    ],
    "Explained Variance": explained_variance,
    "Cumulative Variance": cumulative_variance
})

print("\n" + "=" * 60)
print("PCA EXPLAINED VARIANCE")
print("=" * 60)

print(pca_summary.round(4).to_string(index=False))

# Select components explaining at least 90% variance
n_components = np.argmax(
    cumulative_variance >= 0.90
) + 1

print("\nComponents selected for 90% variance:",
      n_components)

# Transform data using selected components
pca_final = PCA(n_components=n_components)

X_pca_final = pca_final.fit_transform(X_scaled)

print("PCA transformed dataset shape:",
      X_pca_final.shape)

# ==================================================
# 8. K-MEANS CLUSTER EVALUATION
# ==================================================

inertia_values = []
silhouette_values = []

k_values = range(2, 11)

print("\n" + "=" * 60)
print("K-MEANS CLUSTER EVALUATION")
print("=" * 60)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_pca_final)

    inertia_values.append(kmeans.inertia_)

    score = silhouette_score(
        X_pca_final,
        labels,
        sample_size=5000,
        random_state=42
    )

    silhouette_values.append(score)

    print(
        f"K={k} | "
        f"Inertia={kmeans.inertia_:.2f} | "
        f"Silhouette Score={score:.4f}"
    )

    # ==================================================
# 9. FINAL K-MEANS MODEL
# ==================================================

best_k = 4

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = final_kmeans.fit_predict(X_pca_final)

shops_for_clustering["kmeans_cluster"] = kmeans_labels

print("\n" + "=" * 60)
print("FINAL K-MEANS CLUSTER DISTRIBUTION")
print("=" * 60)

print(
    shops_for_clustering["kmeans_cluster"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nCluster percentages:")

print(
    (
        shops_for_clustering["kmeans_cluster"]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2).to_string()
)

# ==================================================
# 10. K-MEANS CLUSTER PROFILES
# ==================================================

profile_features = [
    "avg_monthly_transactions",
    "avg_monthly_ration_cards",
    "transactions_per_ration_card",
    "total_rice",
    "total_transaction_amount",
    "total_sugar"
]

cluster_profile = (
    shops_for_clustering
    .groupby("kmeans_cluster")[profile_features]
    .mean()
    .round(2)
)

print("\n" + "=" * 60)
print("K-MEANS CLUSTER PERFORMANCE PROFILES")
print("=" * 60)

print(
    cluster_profile.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)

# Save cluster profiles
cluster_profile.to_csv(
    OUTPUT_DIR / "kmeans_cluster_profiles.csv"
)

# ==================================================
# 11. INVESTIGATE TRANSACTION AMOUNT
# ==================================================

amount_check = (
    shops_for_clustering
    .groupby("kmeans_cluster")
    .agg(
        shop_count=("shopNo", "count"),
        avg_amount=("total_transaction_amount", "mean"),
        median_amount=("total_transaction_amount", "median"),
        zero_amount_count=(
            "total_transaction_amount",
            lambda x: (x == 0).sum()
        ),
        avg_transactions=("avg_monthly_transactions", "mean")
    )
    .round(2)
)

print("\n" + "=" * 60)
print("TRANSACTION AMOUNT INVESTIGATION")
print("=" * 60)

print(amount_check.to_string())

# ==================================================
# 12. ZERO TRANSACTION AMOUNT DATA QUALITY CHECK
# ==================================================

zero_amount_shops = shops_for_clustering[
    shops_for_clustering["total_transaction_amount"] == 0
].copy()

print("\n" + "=" * 60)
print("ZERO TRANSACTION AMOUNT DATA QUALITY CHECK")
print("=" * 60)

print("\nTotal shops with zero transaction amount:",
      len(zero_amount_shops))

print("\nTransaction activity among zero-amount shops:")

print(
    zero_amount_shops[
        [
            "total_transactions",
            "transaction_months",
            "avg_monthly_transactions"
        ]
    ].describe().round(2)
)

print("\nZero-amount shops by K-Means cluster:")

print(
    zero_amount_shops["kmeans_cluster"]
    .value_counts()
    .sort_index()
)

print("\nZero-amount shops by district:")

print(
    zero_amount_shops["district_name"]
    .value_counts()
    .head(15)
)

print("\nSample zero-amount shops:")

print(
    zero_amount_shops[
        [
            "shopNo",
            "district_name",
            "total_transactions",
            "transaction_months",
            "total_transaction_amount",
            "kmeans_cluster"
        ]
    ].head(10).to_string(index=False)
)
# ==================================================
# 13. DISTRICT-WISE CLUSTER DISTRIBUTION
# ==================================================

district_cluster = pd.crosstab(
    shops_for_clustering["district_name"],
    shops_for_clustering["kmeans_cluster"]
)

district_cluster["total_shops"] = district_cluster.sum(axis=1)

print("\n" + "=" * 60)
print("DISTRICT-WISE K-MEANS CLUSTER DISTRIBUTION")
print("=" * 60)

print(district_cluster.to_string())

# Save district-wise cluster distribution

district_cluster.to_csv(
    OUTPUT_DIR / "district_cluster_distribution.csv"
)


# ==================================================
# 14. DBSCAN - K-DISTANCE ANALYSIS
# ==================================================

import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("DBSCAN K-DISTANCE ANALYSIS")
print("=" * 60)

neighbors = NearestNeighbors(n_neighbors=10)

neighbors.fit(X_pca_final)

distances, indices = neighbors.kneighbors(X_pca_final)

k_distances = np.sort(distances[:, -1])

print("\nSelected k-distance values:")

for percentile in [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]:
    print(
        f"{percentile}%:",
        round(np.percentile(k_distances, percentile), 4)
    )

plt.figure(figsize=(10, 6))

plt.plot(k_distances)

plt.title("DBSCAN - 10th Nearest Neighbor Distance")
plt.xlabel("Data Points (Sorted)")
plt.ylabel("10th Nearest Neighbor Distance")

plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "dbscan_k_distance_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nK-distance plot saved successfully.")


# ==================================================
# 15. DBSCAN PARAMETER COMPARISON
# ==================================================

print("\n" + "=" * 60)
print("DBSCAN PARAMETER COMPARISON")
print("=" * 60)

eps_values = [0.30, 0.35, 0.40, 0.45]

min_samples = 10

for eps in eps_values:

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X_pca_final)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    n_noise = list(labels).count(-1)

    noise_percentage = (n_noise / len(labels)) * 100

    print(
        f"EPS={eps:.2f} | "
        f"Clusters={n_clusters} | "
        f"Noise Points={n_noise} | "
        f"Noise Percentage={noise_percentage:.2f}%"
    )

# ==================================================
# 16. FINAL DBSCAN CLUSTER DISTRIBUTION
# ==================================================

dbscan_final = DBSCAN(
    eps=0.40,
    min_samples=10
)

dbscan_labels = dbscan_final.fit_predict(X_pca_final)

shops_for_clustering["dbscan_cluster"] = dbscan_labels

print("\n" + "=" * 60)
print("FINAL DBSCAN CLUSTER DISTRIBUTION")
print("=" * 60)

print(
    shops_for_clustering["dbscan_cluster"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nCluster percentages:")

print(
    (
        shops_for_clustering["dbscan_cluster"]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2).to_string()
)

# ==================================================
# 17. DBSCAN NOISE SHOP ANALYSIS
# ==================================================

noise_shops = shops_for_clustering[
    shops_for_clustering["dbscan_cluster"] == -1
].copy()

noise_profile = noise_shops[
    [
        "avg_monthly_transactions",
        "avg_monthly_ration_cards",
        "transactions_per_ration_card",
        "total_rice",
        "total_sugar"
    ]
].describe().round(2)

print("\n" + "=" * 60)
print("DBSCAN NOISE SHOP PROFILE")
print("=" * 60)

print(noise_profile.to_string())

print("\nNoise shops by district:")

print(
    noise_shops["district_name"]
    .value_counts()
    .head(15)
    .to_string()
)

# Save noise shop records

noise_shops.to_csv(
    OUTPUT_DIR / "dbscan_noise_shops.csv",
    index=False
)

# ==================================================
# 18. NOISE VS NORMAL SHOP COMPARISON
# ==================================================

comparison_features = [
    "avg_monthly_transactions",
    "avg_monthly_ration_cards",
    "transactions_per_ration_card",
    "total_rice",
    "total_sugar"
]

normal_shops = shops_for_clustering[
    shops_for_clustering["dbscan_cluster"] != -1
]

noise_vs_normal = pd.DataFrame({
    "Noise Shops Mean": noise_shops[comparison_features].mean(),
    "Normal Shops Mean": normal_shops[comparison_features].mean(),
    "Noise Shops Median": noise_shops[comparison_features].median(),
    "Normal Shops Median": normal_shops[comparison_features].median()
}).round(2)

print("\n" + "=" * 60)
print("DBSCAN NOISE VS NORMAL SHOP COMPARISON")
print("=" * 60)

print(noise_vs_normal.to_string())

# District-wise noise percentage

district_noise = (
    shops_for_clustering
    .groupby("district_name")
    .agg(
        total_shops=("shopNo", "count"),
        noise_shops=("dbscan_cluster",
                     lambda x: (x == -1).sum())
    )
)

district_noise["noise_percentage"] = (
    district_noise["noise_shops"] /
    district_noise["total_shops"] * 100
).round(2)

district_noise = district_noise.sort_values(
    "noise_percentage", ascending=False
)

print("\nDistrict-wise DBSCAN noise percentage:")

print(district_noise.to_string())

district_noise.to_csv(
    OUTPUT_DIR / "district_dbscan_noise_analysis.csv"
)

# ==================================================
# 19. SAVE FINAL CLUSTERING OUTPUTS
# ==================================================

print("\n" + "=" * 60)
print("SAVING FINAL CLUSTERING OUTPUTS")
print("=" * 60)

# Complete shop-level clustering results
shops_for_clustering.to_csv(
    OUTPUT_DIR / "shop_clustering_results.csv",
    index=False
)

# Shops without transaction records
shops_without_transactions.to_csv(
    OUTPUT_DIR / "shops_without_transactions.csv",
    index=False
)

# PCA explained variance
pca_summary.to_csv(
    OUTPUT_DIR / "pca_explained_variance.csv",
    index=False
)

# K-Means cluster profiles
cluster_profile.to_csv(
    OUTPUT_DIR / "kmeans_cluster_profiles.csv"
)

# DBSCAN cluster distribution
dbscan_distribution = (
    shops_for_clustering["dbscan_cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("dbscan_cluster")
    .reset_index(name="shop_count")
)

dbscan_distribution.to_csv(
    OUTPUT_DIR / "dbscan_cluster_distribution.csv",
    index=False
)

print("\nClustering output files saved successfully.")

print("\n1. shop_clustering_results.csv")
print("2. shops_without_transactions.csv")
print("3. pca_explained_variance.csv")
print("4. kmeans_cluster_profiles.csv")
print("5. dbscan_cluster_distribution.csv")


# ==================================================
# 20. PCA K-MEANS CLUSTER VISUALIZATION
# ==================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(shops_for_clustering["kmeans_cluster"].unique()):

    cluster_data = X_pca_final[
        shops_for_clustering["kmeans_cluster"].values == cluster
    ]

    plt.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        label=f"Cluster {cluster}",
        alpha=0.5,
        s=12
    )

plt.title("Telangana PDS Shops - K-Means Clusters (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_pca_clusters.png",
    dpi=300
)

plt.show()

print("\nPCA K-Means cluster visualization saved.")

# ==================================================
# 21. K-MEANS CLUSTER SIZE VISUALIZATION
# ==================================================

cluster_counts = (
    shops_for_clustering["kmeans_cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 6))

bars = plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Distribution of Telangana PDS Shops Across K-Means Clusters")
plt.xlabel("K-Means Cluster")
plt.ylabel("Number of Shops")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_cluster_distribution.png",
    dpi=300
)

plt.show()

print("\nK-Means cluster size visualization saved.")

# ==================================================
# 22. DBSCAN CLUSTER DISTRIBUTION VISUALIZATION
# ==================================================

dbscan_counts = (
    shops_for_clustering["dbscan_cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    dbscan_counts.index.astype(str),
    dbscan_counts.values
)

plt.title("DBSCAN Cluster Distribution - Telangana PDS Shops")
plt.xlabel("DBSCAN Cluster (-1 represents Noise)")
plt.ylabel("Number of Shops")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dbscan_cluster_distribution.png",
    dpi=300
)

plt.show()

print("\nDBSCAN cluster distribution visualization saved.")

