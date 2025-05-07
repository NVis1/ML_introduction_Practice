import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from lib import SklearnDataset

from main import safe_log


def main():
    model: Pipeline = joblib.load("model.joblib")
    df = pd.read_csv("./Train.csv").replace([np.inf, -np.inf], np.nan)
    ds = SklearnDataset(
        X=df.drop(columns=["column_5"]),
        y=df["column_5"]
    )

    print(f"\nBest model score: {model.score(ds.X, ds.y)}")

    X = pd.read_csv("./Test.csv").replace([np.inf, -np.inf], np.nan)

    pd.DataFrame(model.predict(X)).to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()
