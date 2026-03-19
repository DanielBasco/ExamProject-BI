import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb NYC – BI Analysis",
    page_icon="🏙️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'DM Serif Display', serif; }

  .metric-card {
    background: #f7f4ef;
    border-left: 4px solid #e05c2a;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
  }
  .metric-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 1.6rem; font-weight: 500; color: #1a1a1a; }

  .result-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
  }
  .result-box .label { font-size: 0.85rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.1em; }
  .result-box .value { font-size: 2.8rem; font-weight: 300; margin: 0.25rem 0; }

  .stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; font-size: 0.95rem; }
  .stTabs [aria-selected="true"] { color: #e05c2a !important; border-bottom-color: #e05c2a !important; }

  section[data-testid="stSidebar"] { background: #1a1a2e; }
  section[data-testid="stSidebar"] * { color: white !important; }
  section[data-testid="stSidebar"] .stSlider > div { color: white; }
</style>
""", unsafe_allow_html=True)


# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("AB_NYC_2019.csv")
    df.drop(columns=["last_review"], inplace=True, errors="ignore")
    df["host_name"] = df["host_name"].fillna("John Doe")

    # Feature engineering
    timesq = (40.758896, -73.985130)
    df["dist_to_cent_km"] = df.apply(
        lambda r: geodesic(timesq, (r["latitude"], r["longitude"])).km, axis=1
    )
    df["num_room_type"] = df["room_type"].map(
        {"Private room": 1, "Entire home/apt": 2, "Shared room": 3}
    )

    # Remove outliers (IQR on price)
    q1, q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
    iqr = q3 - q1
    df = df[(df["price"] >= q1 - 1.5 * iqr) & (df["price"] <= q3 + 1.5 * iqr)]

    # One-hot for neighbourhood_group
    df = pd.get_dummies(df, columns=["neighbourhood_group"], drop_first=True)

    # Price segments for classification
    labels = ["Cheap", "Normal", "Expensive", "Very Expensive"]
    df["price_segment"] = pd.qcut(df["price"], q=4, labels=labels)

    return df


@st.cache_resource
def train_models(df):
    # ── Polynomial Regression ────────────────────────────────────────────────
    X_poly_raw = df[["dist_to_cent_km"]].values
    y_poly = df["price"].values
    poly_transformer = PolynomialFeatures(degree=2)
    X_poly = poly_transformer.fit_transform(X_poly_raw)
    poly_reg = LinearRegression().fit(X_poly, y_poly)

    # ── Random Forest Classifier ─────────────────────────────────────────────
    feat_cols = ["dist_to_cent_km", "minimum_nights", "number_of_reviews",
                 "availability_365", "num_room_type"]
    X_clf = df[feat_cols]
    y_clf = df["price_segment"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=123)
    clf = RandomForestClassifier(random_state=123, n_estimators=100)
    clf.fit(X_tr, y_tr)
    clf_acc = accuracy_score(y_te, clf.predict(X_te))

    # ── KMeans Clustering ────────────────────────────────────────────────────
    cluster_cols = ["dist_to_cent_km", "minimum_nights", "number_of_reviews",
                    "availability_365", "num_room_type", "price"]
    X_km = df[cluster_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_km)
    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    kmeans.fit(X_scaled)
    df = df.copy()
    df["cluster"] = kmeans.labels_

    return poly_transformer, poly_reg, clf, clf_acc, feat_cols, kmeans, scaler, cluster_cols, df


# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models…"):
    df = load_data()
    poly_transformer, poly_reg, clf, clf_acc, feat_cols, kmeans, scaler, cluster_cols, df_clustered = train_models(df)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ Airbnb NYC")
    st.markdown("**BI Exam Project**")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"📋 {len(df):,} listings")
    st.markdown(f"📍 New York City, 2019")
    st.markdown("---")
    st.markdown("**Hypothesis**")
    st.markdown("*H₁: Listings closer to the city center tend to have higher prices.*")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("• Polynomial Regression")
    st.markdown("• Random Forest Classifier")
    st.markdown("• KMeans Clustering")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Airbnb NYC – Price & Distance Analysis")
st.markdown("Does distance from Times Square affect Airbnb prices? Explore the data and test predictions below.")
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Listings</div>
        <div class="metric-value">{len(df):,}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg. Price</div>
        <div class="metric-value">${df['price'].mean():.0f}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg. Distance</div>
        <div class="metric-value">{df['dist_to_cent_km'].mean():.1f} km</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">RF Accuracy</div>
        <div class="metric-value">{clf_acc*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Predictor",
    "🏷️ Segment Classifier",
    "🔵 Cluster Explorer",
    "📊 Data Overview",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Polynomial Regression Price Predictor
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Price Predictor")
    st.markdown("Uses **Polynomial Regression (degree 3)** to predict Airbnb price based on distance to Times Square.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        distance = st.slider(
            "Distance from Times Square (km)",
            min_value=0.0, max_value=40.0, value=5.0, step=0.5
        )
        st.caption(f"Selected: **{distance} km** from Times Square, Midtown Manhattan")

        new_dist = np.array([[distance]])
        new_dist_poly = poly_transformer.transform(new_dist)
        predicted_price = poly_reg.predict(new_dist_poly)[0]
        predicted_price = max(0, predicted_price)

        st.markdown(f"""<div class="result-box">
            <div class="label">Predicted Nightly Price</div>
            <div class="value">${predicted_price:.0f}</div>
            <div class="label">at {distance} km from center</div>
        </div>""", unsafe_allow_html=True)

        # Pearson correlation result
        st.markdown("#### Hypothesis Test")
        st.markdown("""
        | | Result |
        |---|---|
        | Pearson correlation | **-0.389** |
        | P-value | **< 0.001** |
        | Decision | ✅ Reject H₀ |

        A moderate negative correlation confirms that listings **closer to the center cost more**.
        The p-value is far below 0.05, meaning this is **not a coincidence**.
        """)

    with col_right:
        # Plot polynomial curve
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#f7f4ef")
        ax.set_facecolor("#f7f4ef")

        # Sample scatter for performance
        sample = df.sample(min(2000, len(df)), random_state=1)
        ax.scatter(sample["dist_to_cent_km"], sample["price"],
                   alpha=0.15, color="#aaa", s=10, label="Listings")

        X_grid = np.linspace(0, 40, 200).reshape(-1, 1)
        X_grid_poly = poly_transformer.transform(X_grid)
        y_curve = poly_reg.predict(X_grid_poly)
        ax.plot(X_grid, y_curve, color="#e05c2a", linewidth=2.5, label="Poly. Regression")

        ax.axvline(distance, color="#1a1a2e", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axhline(predicted_price, color="#1a1a2e", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.scatter([distance], [predicted_price], color="#e05c2a", s=120, zorder=5)

        ax.set_xlabel("Distance to Times Square (km)", fontsize=10)
        ax.set_ylabel("Price (USD)", fontsize=10)
        ax.set_title("Polynomial Regression: Distance vs. Price", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Random Forest Classifier
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Price Segment Classifier")
    st.markdown("Uses **Random Forest** to classify a listing into: *Cheap, Normal, Expensive, Very Expensive*.")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        dist_clf = st.slider("Distance to Times Square (km)", 0.0, 40.0, 8.0, key="clf_dist")
        min_nights = st.slider("Minimum nights", 1, 30, 2)
        reviews = st.slider("Number of reviews", 0, 300, 25)
        availability = st.slider("Availability (days/year)", 0, 365, 180)
        room_type_clf = st.selectbox("Room type", ["Private room", "Entire home/apt", "Shared room"])

        room_map = {"Private room": 1, "Entire home/apt": 2, "Shared room": 3}
        input_data = pd.DataFrame([[dist_clf, min_nights, reviews, availability, room_map[room_type_clf]]],
                                  columns=feat_cols)

        prediction = clf.predict(input_data)[0]
        probas = clf.predict_proba(input_data)[0]
        classes = clf.classes_

        color_map = {
            "Cheap": "#27ae60",
            "Normal": "#2980b9",
            "Expensive": "#e67e22",
            "Very Expensive": "#c0392b",
        }
        color = color_map.get(str(prediction), "#555")

        st.markdown(f"""<div class="result-box" style="border-top: 4px solid {color};">
            <div class="label">Predicted Segment</div>
            <div class="value" style="color:{color};">{prediction}</div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor("#f7f4ef")
        ax2.set_facecolor("#f7f4ef")
        bar_colors = [color_map.get(str(c), "#999") for c in classes]
        bars = ax2.barh([str(c) for c in classes], probas * 100, color=bar_colors, edgecolor="white")
        ax2.set_xlabel("Probability (%)", fontsize=10)
        ax2.set_title("Classification Confidence", fontsize=11, fontweight="bold")
        ax2.set_xlim(0, 100)
        for bar, p in zip(bars, probas):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{p*100:.1f}%", va="center", fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2)
        plt.close()

        st.markdown(f"**Model accuracy on test set:** `{clf_acc*100:.1f}%`")

        # Feature importance
        fi = pd.DataFrame({"Feature": feat_cols, "Importance": clf.feature_importances_})
        fi = fi.sort_values("Importance", ascending=True)

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        fig3.patch.set_facecolor("#f7f4ef")
        ax3.set_facecolor("#f7f4ef")
        ax3.barh(fi["Feature"], fi["Importance"], color="#1a1a2e")
        ax3.set_title("Feature Importance (Random Forest)", fontsize=10, fontweight="bold")
        ax3.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig3)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – KMeans Cluster Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Cluster Explorer")
    st.markdown("**KMeans (k=4)** segments listings by distance, price, room type, reviews, minimum nights, and availability.")

    # Cluster summary table
    numeric_cols_summary = ["dist_to_cent_km", "minimum_nights", "number_of_reviews",
                            "availability_365", "num_room_type", "price"]
    cluster_summary = df_clustered.groupby("cluster")[numeric_cols_summary].mean().round(1)
    cluster_summary.index = [f"Cluster {i}" for i in cluster_summary.index]
    cluster_summary.columns = ["Dist. (km)", "Min. nights", "# Reviews", "Availability", "Room type", "Avg. price"]

    st.dataframe(cluster_summary.style.background_gradient(cmap="YlOrRd", axis=0), use_container_width=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        fig4.patch.set_facecolor("#f7f4ef")
        ax4.set_facecolor("#f7f4ef")
        sample_c = df_clustered.sample(min(3000, len(df_clustered)), random_state=42)
        scatter = ax4.scatter(
            sample_c["dist_to_cent_km"], sample_c["price"],
            c=sample_c["cluster"], cmap="viridis", alpha=0.4, s=10
        )
        plt.colorbar(scatter, ax=ax4, label="Cluster")
        ax4.set_xlabel("Distance to Center (km)")
        ax4.set_ylabel("Price (USD)")
        ax4.set_title("Clusters: Distance vs. Price", fontweight="bold")
        ax4.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig4)
        plt.close()

    with col_c2:
        # Cluster size bar chart
        cluster_counts = df_clustered["cluster"].value_counts().sort_index()
        fig5, ax5 = plt.subplots(figsize=(6, 4.5))
        fig5.patch.set_facecolor("#f7f4ef")
        ax5.set_facecolor("#f7f4ef")
        ax5.bar([f"Cluster {i}" for i in cluster_counts.index], cluster_counts.values,
                color=["#264653", "#2a9d8f", "#e9c46a", "#e05c2a"])
        ax5.set_ylabel("Number of Listings")
        ax5.set_title("Listings per Cluster", fontweight="bold")
        ax5.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig5)
        plt.close()

    # Predict cluster for new listing
    st.markdown("---")
    st.markdown("### Classify a new listing")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        new_dist_km = st.number_input("Distance (km)", 0.0, 50.0, 10.0)
        new_min_nights = st.number_input("Min. nights", 1, 30, 2)
    with cc2:
        new_reviews = st.number_input("# Reviews", 0, 300, 10)
        new_avail = st.number_input("Availability (days)", 0, 365, 200)
    with cc3:
        new_room = st.selectbox("Room type", ["Private room", "Entire home/apt", "Shared room"], key="cluster_room")
        new_price = st.number_input("Price (USD)", 0, 500, 100)

    new_listing = np.array([[new_dist_km, new_min_nights, new_reviews, new_avail,
                             room_map[new_room], new_price]])
    new_scaled = scaler.transform(new_listing)
    predicted_cluster = kmeans.predict(new_scaled)[0]

    st.markdown(f"""<div class="result-box">
        <div class="label">Predicted Cluster</div>
        <div class="value">Cluster {predicted_cluster}</div>
        <div class="label">{cluster_summary.iloc[predicted_cluster].to_dict()}</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Data Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Data Overview")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        fig6.patch.set_facecolor("#f7f4ef")
        ax6.set_facecolor("#f7f4ef")
        df["price"].hist(bins=50, ax=ax6, color="#e05c2a", edgecolor="white")
        ax6.set_title("Price Distribution", fontweight="bold")
        ax6.set_xlabel("Price (USD)")
        ax6.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig6)
        plt.close()

    with col_d2:
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        fig7.patch.set_facecolor("#f7f4ef")
        df.groupby("room_type")["price"].mean().plot(kind="bar", ax=ax7, color=["#264653", "#2a9d8f", "#e05c2a"],
                                                      edgecolor="white")
        ax7.set_title("Avg. Price by Room Type", fontweight="bold")
        ax7.set_xlabel("")
        ax7.tick_params(axis="x", rotation=20)
        ax7.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig7)
        plt.close()

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        corr_cols = ["price", "minimum_nights", "number_of_reviews", "availability_365", "dist_to_cent_km"]
        corr_matrix = df[corr_cols].corr()
        fig8, ax8 = plt.subplots(figsize=(6, 4.5))
        fig8.patch.set_facecolor("#f7f4ef")
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax8,
                    linewidths=0.5, square=True)
        ax8.set_title("Correlation Heatmap", fontweight="bold")
        st.pyplot(fig8)
        plt.close()

    with col_d4:
        # Neighbourhood group avg price (reconstruct from one-hot)
        orig_df = pd.read_csv("AB_NYC_2019.csv")
        orig_df["dist_to_cent_km"] = df["dist_to_cent_km"].values[:len(orig_df)] if len(orig_df) <= len(df) else 0
        nb_price = orig_df.groupby("neighbourhood_group")["price"].mean().sort_values(ascending=False)
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        fig9.patch.set_facecolor("#f7f4ef")
        nb_price.plot(kind="bar", ax=ax9, color="#1a1a2e", edgecolor="white")
        ax9.set_title("Avg. Price by Borough", fontweight="bold")
        ax9.set_xlabel("")
        ax9.tick_params(axis="x", rotation=20)
        ax9.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig9)
        plt.close()

    st.markdown("### Raw Data Sample")
    display_cols = ["name", "neighbourhood_group", "room_type", "price", "dist_to_cent_km",
                    "minimum_nights", "number_of_reviews", "availability_365"]
    available = [c for c in display_cols if c in df_clustered.columns]
    st.dataframe(df_clustered[available].sample(50, random_state=1).reset_index(drop=True),
                 use_container_width=True)
