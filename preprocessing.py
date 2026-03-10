import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df):

    for col in df.columns:

        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)

        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


def remove_duplicates(df):

    return df.drop_duplicates()


def encode_categorical(df):

    le = LabelEncoder()

    for col in df.select_dtypes(include="object"):

        df[col] = le.fit_transform(df[col])

    return df


def scale_features(df):

    scaler = StandardScaler()

    num_cols = df.select_dtypes(include="number").columns

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def preprocess_pipeline(df):

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = encode_categorical(df)

    df = scale_features(df)

    return df