import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Hierarchical Clustering App")

st.title("🔗 Hierarchical Clustering")
st.write("Agglomerative Clustering Model")

# Load model
with open("Hierarchical_Cluster.pkl", "rb") as file:
    model = pickle.load(file)

st.subheader("Enter Data Point")

# Input features (change labels if needed)
feature1 = st.number_input("Feature 1", step=0.1)
feature2 = st.number_input("Feature 2", step=0.1)

if st.button("Find Cluster"):
    X = np.array([[feature1, feature2]])

    # Hierarchical clustering does NOT support predict()
    try:
        cluster = model.fit_predict(X)
        st.success(f"🧩 Assigned Cluster: {cluster[0]}")
    except Exception as e:
        st.error("This model cannot assign clusters to a single new point.")
        st.info("Hierarchical clustering is mainly for analyzing existing datasets.")
