def understand_query(query):

    query = query.lower()

    if "visual" in query or "plot" in query:
        return "visualization"

    elif "train" in query or "model" in query:
        return "ml"

    elif "summary" in query or "eda" in query:
        return "eda"

    elif "insight" in query:
        return "insight"

    else:
        return "unknown"