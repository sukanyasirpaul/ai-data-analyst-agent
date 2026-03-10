import pandas as pd

def load_data(file):

    filename = file.name

    if filename.endswith(".csv"):
        df = pd.read_csv(file)

    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file)

    elif filename.endswith(".json"):
        df = pd.read_json(file)

    else:
        raise ValueError("Unsupported file type")

    return df