import os

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from lib import SklearnDataset

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier

from summarizing_practical_task.model_handler import ModelHandler

import logging


def old_get_feature_importance_for(model, transformer):
    return pd.DataFrame({
        "Feature": transformer.get_feature_names_out(),
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)


def main(df: pd.DataFrame, models, remove_features: list[str] = None, include_features: list[str] = None):
    dataset = create_dataset(df, y_col_name="Churn",
                             remove_features=remove_features, include_features=include_features)

    handler = ModelHandler(*dataset.target_tt_split())
    handler.make_column_transformer(dataset.X)

    results = []
    for k, v in models.items():
        print(f"\n==========MODEL TEST: {k}==========")
        pl = handler.handle(v["model"], v["param_grid"])
        results.append({
            "name": k,
            "model": pl,
            "score": pl.score(*handler.test)
        })

    print("\n==========RESULTS==========\n")
    for index, i in enumerate(results):
        print(f"\t[{index+1}] Model: {i['name']} best cross-validation score: {i['score']*100}%;")
    print("\n")

    choose_and_save(results)


def create_dataset(df, y_col_name: str, remove_features: list[str], include_features: list[str]):
    # print(df.dtypes)

    # Fixes error with TotalCharges column being incorrectly identified as categorical.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df[y_col_name] = LabelEncoder().fit_transform(df[y_col_name])

    return SklearnDataset.from_df(
        df, y_col_name=y_col_name,
        remove_features=remove_features,
        include_features=include_features
    )


def choose_and_save(results):
    print("Which one to save? (0 if none)")
    while True:
        choice: str = input("> ")
        if not choice.isdigit():
            print("Please enter a number corresponding to model you want to save, or 0 if none.")
            continue

        choice: int = int(choice) - 1
        if not -1 < choice < len(results):
            print("Incorrect number provided.")
            continue

        if choice == -1:
            print("No model was saved.")
        else:
            newest = "model-newest.joblib"
            if os.path.exists(newest):
                os.rename(newest, get_next_model_version_name())

            joblib.dump(results[choice]["model"], newest)
            print("Model was saved.")
        break


def get_next_model_version_name():
    """
    Finds the next available model version based on existing files.
    Notice: This code is CHATGPT-SUGGESTED for greater convenience in saving versions and may not be the best.
    Revise and rewrite later.
    """
    import re
    files = os.listdir()
    pattern = re.compile(r"model-v(\d+)\.joblib")

    versions = [int(match.group(1)) for f in files if (match := pattern.match(f))]
    v = max(versions, default=0) + 1  # Increment from the highest version

    return f"model-v{v}.joblib"


def models_for_gridsearch():
    return {
        "XGBoostRandomForest": {
            "model": XGBRFClassifier(
                random_state=42, verbosity=0, warm_start=True),
            "param_grid": {
                "n_estimators": [42, 200, 500],
                "max_features": ["auto", 5, 10, 20, 42],
                "min_samples_leaf": [4, 2],
                "max_depth": [4, 6, 8],
                "criterion": ["gini", "entropy"]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, verbosity=0, warm_start=True, booster="gblinear"),
            "param_grid": {
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, verbose=0, warm_start=True),
            "param_grid": {
                "n_estimators": [42, 200, 500],
                "max_features": ["auto", "sqrt", "log2", 5, 10, 20],
                "min_samples_leaf": [1, 3, 5],
                "max_depth": [4, 6, 8],
                "criterion": ["gini", "entropy"]
            }
        },
    }


if __name__ == '__main__':
    main(
        df=pd.read_csv(
            "../datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        ).sample(frac=0.5, random_state=42),
        models=models_for_gridsearch(),
        remove_features=["customerID"],
        include_features=["InternetService", "Contract", "tenure", "TotalCharges"]  # identified via feature importance
    )
