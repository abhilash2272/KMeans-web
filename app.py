import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.markdown("""
<style>
body{background-color:#0e1117}
h1,h2,h3{font-family:Segoe UI,sans-serif}
.note{background:#102a43;padding:12px;border-radius:8px;color:#e6f1ff}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🟢 Customer Segmentation Dashboard")
st.write("This system uses K-Means Clustering to group customers based on their purchasing behavior and similarities.")

df = pd.read_csv("Wholesale customers data.csv")
data = df.drop(columns=["Channel", "Region"])

st.sidebar.header("Clustering Controls")

feature_1 = st.sidebar.selectbox("Select Feature 1", data.columns)
feature_2 = st.sidebar.selectbox("Select Feature 2", data.columns, index=1)
k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)
random_state = st.sidebar.number_input("Random State", value=42, step=1)
run = st.sidebar.button("🟦 Run Clustering")

if run:
    X = data[[feature_1, feature_2]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_result = df.copy()
    df_result["Cluster"] = clusters

    st.markdown("### 📊 Cluster Visualization")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis", alpha=0.75)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", s=140, marker="X")
        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        counts = df_result["Cluster"].value_counts().sort_index()
        means = df_result.groupby("Cluster")[[feature_1, feature_2]].mean()

        summary = pd.DataFrame({
            "Cluster Size": counts,
            f"Avg {feature_1}": means[feature_1].round(2),
            f"Avg {feature_2}": means[feature_2].round(2)
        })

        st.markdown("### 📋 Cluster Summary")
        st.dataframe(summary, use_container_width=True)

    st.markdown("### 💡 Business Interpretation")

    mean_1 = X[feature_1].mean()
    mean_2 = X[feature_2].mean()

    for c in summary.index:
        avg_1 = summary.loc[c, f"Avg {feature_1}"]
        avg_2 = summary.loc[c, f"Avg {feature_2}"]

        if avg_1 > mean_1 and avg_2 > mean_2:
            st.success(f"Cluster {c}: High-spending customers across selected categories.")
        elif avg_1 < mean_1 and avg_2 < mean_2:
            st.warning(f"Cluster {c}: Budget-conscious customers with lower spending.")
        else:
            st.info(f"Cluster {c}: Moderate spenders with selective purchasing behavior.")

    st.markdown("<div class='note'>📌 Customers in the same cluster exhibit similar purchasing behaviour and can be targeted with similar business strategies.</div>", unsafe_allow_html=True)
