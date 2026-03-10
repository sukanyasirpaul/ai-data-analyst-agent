import streamlit as st
import pandas as pd

from agents.data_loader import load_data
from agents.eda_agent import dataset_overview, statistical_summary
from agents.visualization_agent import (
    correlation_heatmap,
    histogram_plot,
    box_plot,
    scatter_plot,
    distribution_plot
)
from agents.insight_agent import generate_insights
from agents.feature_engineering_agent import encode_categorical
from agents.ml_agent import auto_ml
from utils.preprocessing import preprocess_pipeline
from utils.report_generator import generate_report


st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("🤖 AI Data Analyst Agent")
st.write("Upload a dataset and let AI analyze it automatically.")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv","xlsx","json"])

if uploaded_file:

    df = load_data(uploaded_file)

    # preprocessing
    df = preprocess_pipeline(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    option = st.sidebar.selectbox(
        "Choose Analysis",
        ["EDA", "Visualization", "Insights", "Machine Learning", "Generate Report"]
    )

# ---------------- EDA ----------------

    if option == "EDA":

        st.subheader("Dataset Overview")

        overview = dataset_overview(df)
        st.write(overview)

        st.subheader("Statistical Summary")
        st.write(statistical_summary(df))


# ---------------- VISUALIZATION ----------------

    elif option == "Visualization":

        st.subheader("📊 Smart Visualization Dashboard")

        numeric_cols = df.select_dtypes(include="number").columns

        important_cols = (
            df[numeric_cols]
            .var()
            .sort_values(ascending=False)
            .head(5)
            .index
        )

        st.write("Important Columns Detected:", list(important_cols))

        st.subheader("Correlation Heatmap")
        correlation_heatmap(df)

        st.subheader("Histograms")

        for col in important_cols:
            histogram_plot(df, col)

        st.subheader("Box Plots")

        for col in important_cols:
            box_plot(df, col)

        st.subheader("Scatter Plots")

        if len(important_cols) >= 2:
            scatter_plot(df, important_cols[0], important_cols[1])

        if len(important_cols) >= 3:
            scatter_plot(df, important_cols[1], important_cols[2])

        st.subheader("Distribution Plots")

        for col in important_cols[:3]:
            distribution_plot(df, col)


# ---------------- INSIGHTS ----------------

    elif option == "Insights":

        st.subheader("Auto Generated Insights")

        insights = generate_insights(df)

        for i in insights:
            st.write(i)


# ---------------- MACHINE LEARNING ----------------

    elif option == "Machine Learning":

        st.subheader("🤖 Automatic Machine Learning")

        df_encoded = encode_categorical(df)

        potential_targets = []

        for col in df.columns:

            if df[col].nunique() <= 10:
                potential_targets.append(col)

        if len(potential_targets) > 0:
            target = potential_targets[0]
        else:
            target = df.columns[-1]

        st.write("Detected Target Column:", target)

        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]

        st.write("Training Multiple Models...")

        best_model, results = auto_ml(X, y)

        st.subheader("Model Performance")

        results_df = pd.DataFrame(
            list(results.items()),
            columns=["Model", "Score"]
        )

        st.dataframe(results_df)

        st.success(f"Best Model Selected: {best_model}")


# ---------------- REPORT ----------------

    elif option == "Generate Report":

        st.subheader("📄 AI Generated Report")

        generate_report(df)