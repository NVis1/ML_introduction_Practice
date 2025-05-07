import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold

from imblearn.pipeline import Pipeline as ImbPipeline, Pipeline

from Task.column_selector import ColumnSelector
from lib import BetterPipeline, SklearnDataset

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


from Task.data_preparation import prepare_dataframe, get_not_high_corr_columns
from Task.training import train


random_state = 42

payloads = [
        # "name": "RF",
    {
        "name": "XGB",
        "estimator": XGBClassifier(
            random_state=random_state,
            max_depth=8,
            max_leaves=40,
            n_estimators=66,
        ),
        "param_grid": {
            # "n_estimators": [66, 68],
            # "max_leaves": [38, 40, 42],
            # "max_depth": [8, 10, 12]
        }
    },
    # {
    #     "name": "SVC",
    #     "estimator": SVC(
    #         random_state=random_state,
    #         C=42,
    #         kernel="rbf",
    #         gamma="auto",
    #         degree=4,
    #         coef0=1.0,
    #     ),
    #     "param_grid": {
    #         "C": [0, 0.1, 1, 42, 100],
    #         "kernel": ["linear", "rbf"],
    #         "coef0": [0.42, 1.0, 1.42, 2.0],
    #         "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10],
    #         "degree": [2, 3, 4, 5],
    #     },
    # },
]


def safe_log(X):
    return np.log( np.clip( X, 1e-5, None ) )


preprocessor = ImbPipeline(list({
    "NanImputer":   SimpleImputer(strategy="median"),
    "InfImputer":   SimpleImputer(strategy="median", missing_values=np.inf),
    "-InfImputer":   SimpleImputer(strategy="median", missing_values=-np.inf),
    "LogTransformer": FunctionTransformer(safe_log),
    "StdScaler":    MinMaxScaler(),
    # "VarianceSelector": VarianceThreshold(threshold=0.01),
    "Smote": SMOTE(random_state=random_state),
}.items()))


def main():
    print("Preparing...")
    df = prepare_dataframe()
    ds = SklearnDataset.from_df(df, y_col_name="column_5")
    # .sample(42424)
    ds_full = SklearnDataset.from_df(df, y_col_name="column_5")

    add_high_corr_columns_filter_to(preprocessor, ds)

    print("Training...")
    training_result = train(
        ds=ds,
        payloads=payloads,
        preprocessor=preprocessor
    )

    print("Training complete.")
    pipeline = find_best(training_result)

    print("Computing score on the full dataset...")

    score = f1_score(ds_full.y, pipeline.predict(ds_full.X), average="weighted")
    print(f"\nBest model metrics: {100*score:.4f}")

    print("Saving best model...")

    joblib.dump(pipeline, "model.joblib")
    print("Model saved.")


def add_high_corr_columns_filter_to(preprocessor, ds):
    corr_col_selector = ColumnSelector(
        selected_columns=get_not_high_corr_columns(ds)
    )
    preprocessor.steps.insert(
        0, ("ColumnSelector", corr_col_selector))


def find_best(res):
    current_best = 0
    best = None

    for i in res:

        print(
            f"\nModel { i["name"] } results: \n"
            f"\tBest params: { i["searcher"].best_params_ } \n"
            f"\tBest score: { (100 * i["searcher"].best_score_):.3f} \n"
        )

        if current_best < i["searcher"].best_score_:
            current_best = i["searcher"].best_score_
            best = i["searcher"].best_estimator_

    return best


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()
