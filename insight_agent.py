import pandas as pd


def generate_insights(df):

    insights = []

    # Dataset info
    rows, cols = df.shape
    insights.append(f"Dataset contains {rows} rows and {cols} columns.")

    # Missing values
    missing = df.isnull().sum().sum()

    if missing == 0:
        insights.append("Dataset has no missing values.")
    else:
        insights.append(f"Dataset contains {missing} missing values.")

    # Numeric columns
    numeric_cols = df.select_dtypes(include="number").columns

    # High variance features
    variance = df[numeric_cols].var().sort_values(ascending=False)

    top_var = variance.head(3).index.tolist()

    insights.append(
        f"Features with highest variation: {', '.join(top_var)}."
    )

    # Correlation insights
    corr = df[numeric_cols].corr().abs()

    high_corr_pairs = []

    for i in range(len(corr.columns)):
        for j in range(i):

            if corr.iloc[i, j] > 0.8:

                col1 = corr.columns[i]
                col2 = corr.columns[j]

                high_corr_pairs.append(f"{col1} & {col2}")

    if high_corr_pairs:
        insights.append(
            f"Highly correlated feature pairs detected: {', '.join(high_corr_pairs[:3])}."
        )

    # Outlier detection
    outlier_cols = []

    for col in numeric_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]

        if len(outliers) > 0:

            outlier_cols.append(col)

    if outlier_cols:

        insights.append(
            f"Potential outliers detected in: {', '.join(outlier_cols[:3])}."
        )

    # Feature importance hint
    if len(numeric_cols) > 5:

        insights.append(
            "Dataset contains multiple numeric features suitable for machine learning models."
        )

    # Data quality
    insights.append(
        "Dataset appears structured and suitable for further machine learning analysis."
    )

    return insights