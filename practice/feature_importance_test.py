import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np

from lib import SklearnDataset, make_standart_preprocessor_for, BetterPipeline


def handle_estimator(estimator: BaseEstimator, X, y):
    print(f"[Debug] Model score: {estimator.score(X, y)}")

    # Compute permutation importance
    result = permutation_importance(
        estimator, X, y,
        scoring="accuracy", n_repeats=10, random_state=42
    ).importances_mean

    # Show importance
    importances = pd.Series(
        data=result,
        index=X.columns)
    return importances.sort_values(ascending=False)


def main(Xy):
    dataset = SklearnDataset(*Xy)
    train, test = dataset.target_tt_split()

    train = SklearnDataset(train.X, train.y.squeeze().astype('category').cat.codes)
    test  = SklearnDataset(test.X,   test.y.squeeze().astype('category').cat.codes)

    print("Training raw (untransformed) model...")

    model_raw = xgb.XGBClassifier(n_estimators=42, max_depth=42, learning_rate=0.042,
                                  enable_categorical=True)
    model_raw.fit(*train)

    output1 = str(handle_estimator(model_raw, *test))

    print("Training pipeline (transformed model)...")

    pipeline = BetterPipeline(
        steps=dict(
            preprocessor=make_standart_preprocessor_for(train.X),
            model=xgb.XGBClassifier(n_estimators=42, max_depth=42, learning_rate=0.042),
        ),
        memory=None
    )
    pipeline.fit_try_transform(*train)

    output2 = str(handle_estimator(pipeline, *test))

    print(output1, end="\n\n\n")
    print(output2, end="\n\n\n")
    return output1 == output2


if __name__ == '__main__':
    from sklearn.datasets import load_wine, fetch_openml
    # df = load_wine(return_X_y=True)
    df = fetch_openml(name="adult", version=2, return_X_y=True)

    res = main(df)
    print(f"Feature importance equality: {res}")
