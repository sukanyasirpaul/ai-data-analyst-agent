from sklearn.preprocessing import LabelEncoder

def encode_categorical(df):

    le = LabelEncoder()

    for col in df.select_dtypes(include="object"):

        df[col] = le.fit_transform(df[col])

    return df