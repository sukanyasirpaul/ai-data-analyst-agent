import pandas as pd

def dataset_overview(df):

    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

    return overview


def statistical_summary(df):

    return df.describe()


def correlation_matrix(df):

    return df.corr(numeric_only=True)


def detect_outliers(df):

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    IQR = Q3 - Q1

    outliers = ((df < (Q1 - 1.5 * IQR)) |
                (df > (Q3 + 1.5 * IQR))).sum()

    return outliers