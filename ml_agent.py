import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from sklearn.metrics import accuracy_score, r2_score

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import matplotlib.pyplot as plt


def auto_ml(X, y):

    # encode categorical target
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # detect target type using sklearn
    target_type = type_of_target(y)

    if target_type in ["binary", "multiclass"]:
        problem_type = "classification"
    else:
        problem_type = "regression"

    st.write(f"Detected Problem Type: **{problem_type.upper()}**")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # ---------------- CLASSIFICATION ----------------

    if problem_type == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)

            results[name] = score

    # ---------------- REGRESSION ----------------

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "SVR": SVR()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            results[name] = score

    # results dataframe
    results_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Score": list(results.values())
    })

    st.subheader("Model Performance")
    st.dataframe(results_df)

    best_model_name = results_df.loc[results_df["Score"].idxmax(), "Model"]

    st.success(f"Best Model Selected: {best_model_name}")

    # train best model
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    # feature importance (if available)
    if hasattr(best_model, "feature_importances_"):

        st.subheader("Feature Importance")

        importance = best_model.feature_importances_

        fig, ax = plt.subplots()

        ax.barh(X.columns, importance)

        ax.set_title("Feature Importance")

        st.pyplot(fig)

    return best_model_name, results_df