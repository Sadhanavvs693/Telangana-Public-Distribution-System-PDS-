# Telangana PDS Data Analytics, Anomaly Detection & Machine Learning

## 📌 Project Overview

This project analyzes **Telangana Public Distribution System (PDS)** data to understand ration distribution patterns, identify unusual Fair Price Shop (FPS) activity, and develop machine learning models to predict high transaction activity for the following month.

The project combines data analytics, statistical methods, unsupervised learning, supervised machine learning, and interactive dashboard development.

### Key Components

- Data cleaning and integration
- Exploratory Data Analysis (EDA)
- District and shop-level analysis
- Statistical anomaly detection using IQR
- Isolation Forest
- Multi-method anomaly investigation
- Data leakage investigation
- Temporal feature engineering
- Machine learning classification
- PCA-based dimensionality reduction
- K-Means and DBSCAN clustering
- Interactive Streamlit dashboard

---

## 🎯 Business Problem

The Telangana PDS system contains transaction, ration-card, commodity distribution, and Fair Price Shop information across multiple districts and months.

This project addresses the following business questions:

1. Which districts and shops have the highest transaction activity?
2. Which shops show unusual transaction patterns?
3. Can multiple anomaly-detection methods identify high-priority shops for further investigation?
4. Can historical transaction behavior predict high transaction activity in the following month?
5. Which factors are useful for predicting future transaction activity?
6. Can shops be grouped into meaningful operational activity segments?
7. How can analytical results be presented through an interactive dashboard?

> **Important:** Anomalies identified in this project are investigation signals. They do not automatically indicate fraud or wrongdoing.

---

## 📊 Dataset

The project integrates three major sources of PDS information.

### 1. Card Status Data

Contains information related to ration cards, units, card categories, districts, shops, and monthly records.

### 2. Transaction Data

Contains monthly transaction and commodity distribution information for Fair Price Shops.

### 3. FPS Location Data

Contains Fair Price Shop information, including:

- District
- Shop number
- Address
- Latitude and longitude
- FPS status
- FPS type

These datasets were cleaned, processed, and integrated into a master dataset.

### Master Dataset

| Description | Value |
|---|---:|
| Total records | 517,666 |
| Total columns | 53 |
| Districts | 33 |
| Fair Price Shops | 17,576 |
| Shops with transaction activity | 17,491 |
| Shops without transaction records | 85 |

---

## 🔍 Exploratory Data Analysis

EDA was performed to understand the distribution of transactions, ration cards, commodity quantities, and shop-level activity.

Key areas included:

- District-wise transaction distribution
- Shop-level transaction patterns
- Monthly transaction trends
- Ration-card distribution
- Commodity distribution, including rice and sugar
- Transaction-to-ration-card ratios
- Identification of missing and unusual values
- Data quality investigation

---

## 🚨 Anomaly Detection

Multiple approaches were used to identify unusual shop-level transaction patterns.

### 1. IQR-Based Statistical Detection

The Interquartile Range method was applied to identify unusually high values.

| Metric | Result |
|---|---:|
| Q1 | 276.77 |
| Q3 | 536.54 |
| IQR | 259.77 |
| Upper threshold | 926.19 |
| Transaction outliers | 910 |
| Transaction/ration-card ratio outliers | 958 |
| Rice-per-transaction outliers | 267 |

### 2. Isolation Forest

Isolation Forest was used to identify unusual shop activity using multiple numerical features.

**Model configuration:**
- Algorithm: Isolation Forest
- Estimators: 200
- Contamination: 0.05
- Random state: 42

**Results:**

| Category | Shops |
|---|---:|
| Normal | 16,616 |
| Anomalous | 875 |
| Total | 17,491 |

### 3. Multi-Method Anomaly Investigation

Results from statistical outlier detection and Isolation Forest were compared to identify shops flagged by multiple methods.

Key findings:

- 627 shops flagged by transaction outlier detection and Isolation Forest.
- 383 shops flagged by ratio outlier detection and Isolation Forest.
- 74 shops flagged by rice outlier detection and Isolation Forest.
- 781 shops identified through the common IQR and Isolation Forest comparison.
- 296 shops classified as high-priority investigation candidates.
- 7 shops flagged by all four methods, located in Hyderabad.

These results help prioritize further investigation rather than establish misconduct.

---

## 🤖 Machine Learning – Future Transaction Prediction

An initial classification experiment produced unusually high performance. A leakage investigation identified a strong relationship between current-month features and the target.

For example, the correlation between `noOfRcs` and `noOfTrans` was approximately 0.999804.

To make the prediction task more meaningful, the project was redesigned to predict **high transaction activity in the following month** using historical information.

### Model Development

The workflow included:

1. Temporal feature engineering
2. Target creation for future-month activity
3. Chronological train-test splitting
4. Model training and comparison
5. Model evaluation

### Dataset and Split

| Description | Records |
|---|---:|
| Final modelling dataset | 412,665 |
| Normal activity | 308,714 (74.81%) |
| High transaction activity | 103,951 (25.19%) |
| Training data (2023–2024) | 343,725 |
| Testing data (2025) | 68,940 |

### Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

### Final Model Performance

**Random Forest**

| Metric | Result |
|---|---:|
| Accuracy | 96.79% |
| F1-score | 93.82% |

The chronological split helps evaluate performance on a later period rather than randomly mixing past and future observations.

---

## 🧩 Shop Clustering Analysis

Unsupervised learning was used to group Fair Price Shops based on operational activity patterns.

### Clustering Population

- 17,491 shops with transaction activity were included.
- 85 shops without transaction records were analysed separately.

### Features Used

- Average monthly transactions
- Average monthly ration cards
- Transactions per ration card
- Total rice
- Total sugar

The total transaction amount was excluded from clustering inputs because 9,803 shops had zero recorded transaction amount despite having positive transaction counts. It was retained for diagnostic interpretation.

### Data Preparation

- Log transformation using `log1p` for four skewed features
- StandardScaler for feature standardization
- Principal Component Analysis (PCA)

### PCA Results

Three principal components explained **97.68% of the variance**.

| Component | Explained Variance |
|---|---:|
| PC1 | 62.91% |
| PC2 | 18.72% |
| PC3 | 16.05% |
| PC4 | 2.13% |
| PC5 | 0.18% |

### K-Means Clustering

K-Means was evaluated for cluster counts from 2 to 10.

- Selected cluster count: 4
- Best tested silhouette score: 0.3362
- Random state: 42
- Initializations: 10

| Cluster | Shops | Percentage |
|---|---:|---:|
| 0 | 4,249 | 24.29% |
| 1 | 7,047 | 40.29% |
| 2 | 2,176 | 12.44% |
| 3 | 4,019 | 22.98% |

**Interpretation:**

- Cluster 2: Highest average transaction activity.
- Cluster 3: Lowest average transaction activity.
- Clusters 0 and 1: Moderate activity profiles.

These are data-driven operational segments, not official performance ratings.

### DBSCAN Clustering

DBSCAN was also explored to identify density-based groups and potential outliers.

**Final configuration:**
- Epsilon: 0.40
- Minimum samples: 10

| Category | Shops | Percentage |
|---|---:|---:|
| Noise (-1) | 753 | 4.31% |
| Cluster 0 | 16,689 | 95.41% |
| Cluster 1 | 17 | 0.10% |
| Cluster 2 | 15 | 0.09% |
| Cluster 3 | 17 | 0.10% |

DBSCAN produced one dominant cluster and several very small clusters. Therefore, K-Means was more useful for the primary shop segmentation, while DBSCAN was retained as a supplementary density and outlier exploration method.

Noise points are not automatically poor-performing or fraudulent shops.

### Data Quality Observation

A notable issue was identified: 9,803 shops had zero recorded transaction amount despite positive transaction counts.

This requires further investigation into the underlying source data and should not be assigned an assumed cause.

---

## 📊 Streamlit Dashboard

An interactive Streamlit dashboard was developed to present the analytical findings.

### Dashboard Pages

1. 🏠 Overview
2. 📊 District Analysis
3. 📈 Transaction Analysis
4. 🚨 Anomaly Detection
5. 🤖 ML Prediction
6. 🔍 Shop Investigation
7. 💡 Business Insights
8. 🧩 Shop Clustering

The dashboard brings together district-level insights, shop-level investigation, anomaly results, machine learning outputs, and clustering analysis.

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn, XGBoost |
| Dimensionality Reduction | PCA |
| Clustering | K-Means, DBSCAN |
| Anomaly Detection | IQR, Isolation Forest |
| Dashboard | Streamlit |
| File Handling | OpenPyXL |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

---

## 📁 Repository Structure

The main project components include:

```text
Telangana-PDS-Data-Science/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── shop_analysis.py
│   └── clustering_analysis.py
│
├── data_processed/
│   └── pds_master_dataset.csv   (not included in GitHub)
│
└── output/
    ├── shop_summary.csv         (not included in GitHub)
    ├── clustering results
    └── analysis visualizations
```

Additional project scripts and output files may be present in the repository.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- VS Code or another Python IDE

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sadhanavvs693/Telangana-PDS-Data-Science.git
```

```bash
cd Telangana-PDS-Data-Science
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Required Data

**Important: The original source datasets and some generated CSV files are not included in this public repository.**

The master dataset is excluded from GitHub through `.gitignore`. Consequently, cloning the repository alone does not provide all the data required to reproduce the complete analysis or run every dashboard page.

To reproduce the project:

1. Obtain the original Card Status, Transaction, and FPS Location datasets from their authorized sources.
2. Run the corresponding data cleaning and integration workflow to generate the master dataset.
3. Place the generated master dataset at:

```text
data_processed/pds_master_dataset.csv
```

4. Generate the required shop-level summaries and clustering outputs using the project analysis scripts.
5. Ensure the generated files are available in their expected `output/` locations before running the dashboard.

The required source data and generated outputs must be prepared locally; they are not supplied as part of the GitHub clone.

### Step 5: Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

## 💡 Business Value

This project demonstrates how data analytics and machine learning can support public distribution system monitoring.

Potential applications include:

- Identifying districts with unusual transaction patterns.
- Prioritizing shops for further investigation.
- Understanding differences in shop-level operational activity.
- Predicting future high transaction activity.
- Exploring shop segments based on historical distribution patterns.
- Supporting data-driven monitoring and planning.

These outputs are analytical decision-support tools and should be interpreted alongside operational context and source-data validation.

---

## 🏁 Conclusion

The Telangana PDS analytics project combines large-scale data integration, exploratory analysis, statistical anomaly detection, machine learning, and unsupervised clustering.

The project also demonstrates the importance of investigating data leakage, selecting appropriate validation strategies for temporal prediction, and interpreting anomaly-detection results carefully.

An interactive eight-page Streamlit dashboard brings the findings together in a single analytical interface.

**The overall objective is to transform complex PDS data into meaningful insights that can support better monitoring, investigation, and operational decision-making.**

---