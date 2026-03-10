import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# 1️⃣ Correlation Heatmap
def correlation_heatmap(df):

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)

    st.pyplot(fig)


# 2️⃣ Histogram
def histogram_plot(df, column):

    st.subheader(f"Histogram of {column}")

    fig, ax = plt.subplots()

    sns.histplot(df[column], kde=True, ax=ax)

    st.pyplot(fig)


# 3️⃣ Box Plot
def box_plot(df, column):

    st.subheader(f"Box Plot of {column}")

    fig, ax = plt.subplots()

    sns.boxplot(x=df[column], ax=ax)

    st.pyplot(fig)


# 4️⃣ Scatter Plot
def scatter_plot(df, x_col, y_col):

    st.subheader(f"Scatter Plot: {x_col} vs {y_col}")

    fig, ax = plt.subplots()

    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)

    st.pyplot(fig)


# 5️⃣ Bar Chart
def bar_chart(df, column):

    st.subheader(f"Bar Chart of {column}")

    counts = df[column].value_counts()

    fig, ax = plt.subplots()

    counts.plot(kind="bar", ax=ax)

    st.pyplot(fig)


# 6️⃣ Line Chart
def line_chart(df, x_col, y_col):

    st.subheader(f"Line Chart: {y_col} over {x_col}")

    fig, ax = plt.subplots()

    ax.plot(df[x_col], df[y_col])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    st.pyplot(fig)


# 7️⃣ Pie Chart
def pie_chart(df, column):

    st.subheader(f"Pie Chart of {column}")

    counts = df[column].value_counts()

    fig, ax = plt.subplots()

    ax.pie(counts, labels=counts.index, autopct="%1.1f%%")

    st.pyplot(fig)


# 8️⃣ Pair Plot
def pair_plot(df):

    st.subheader("Pair Plot")

    fig = sns.pairplot(df)

    st.pyplot(fig)


# 9️⃣ Distribution Plot
def distribution_plot(df, column):

    st.subheader(f"Distribution Plot: {column}")

    fig, ax = plt.subplots()

    sns.kdeplot(df[column], fill=True, ax=ax)

    st.pyplot(fig)