import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_report(df):

    st.title("📊 AI Data Analysis Report")

    # ---------------- DATASET OVERVIEW ----------------

    st.header("Dataset Overview")

    rows, cols = df.shape

    col1, col2 = st.columns(2)

    col1.metric("Total Rows", rows)
    col2.metric("Total Columns", cols)

    st.write("### Columns in Dataset")

    st.write(list(df.columns))

    # ---------------- MISSING VALUES ----------------

    st.header("Missing Values Analysis")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(missing_df)

    if missing.sum() == 0:
        st.success("No Missing Values Found")

    # ---------------- STATISTICS ----------------

    st.header("Statistical Summary")

    st.dataframe(df.describe())

    # ---------------- CORRELATION HEATMAP ----------------

    st.header("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    # ---------------- KEY INSIGHTS ----------------

    st.header("AI Generated Insights")

    insights = []

    if missing.sum() == 0:
        insights.append("Dataset is clean with no missing values.")

    if df.shape[1] > 20:
        insights.append("Dataset has many features which may affect model performance.")

    if df.corr().abs().max().max() > 0.8:
        insights.append("Some features are highly correlated.")

    if len(insights) == 0:
        insights.append("Dataset looks balanced and usable for machine learning.")

    for i in insights:
        st.write("•", i)

    # ---------------- DISTRIBUTION PLOTS ----------------

    st.header("Feature Distributions")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols[:5]:

        fig, ax = plt.subplots()

        sns.histplot(df[col], kde=True, ax=ax)

        ax.set_title(col)

        st.pyplot(fig)