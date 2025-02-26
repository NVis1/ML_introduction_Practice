# https://youtu.be/wAVgNAbbT38?si=8nupBf2ihs6i1vDt


import xgboost as xgb
from sklearn.pipeline import Pipeline

from lib import Dataset


def main():
    dataset = Dataset()
    train, test = dataset.target_tt_split()

    pipeline = Pipeline([
        ("model", xgb.XGBClassifier())
    ], memory=None)

    pipeline.fit(*train)

    print(test.mae_percentage(pipeline))


if __name__ == '__main__':
    main()
