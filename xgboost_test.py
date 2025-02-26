# https://youtu.be/wAVgNAbbT38?si=8nupBf2ihs6i1vDt


import xgboost as xgb
from sklearn.pipeline import Pipeline

from lib import Dataset, make_standart_preprocessor_for

from warnings import warn


def main():
    from sklearn.datasets import load_wine
    dataset = Dataset(*load_wine(return_X_y=True))
    train, test = dataset.target_tt_split()

    steps = {
        "preprocessor": make_standart_preprocessor_for(train.X),
        "model": xgb.XGBClassifier(n_estimators=42, max_depth=42, learning_rate=0.042),
    }

    pipeline = Pipeline(list(steps.items()), memory=None)
    try:
        pipeline.fit_transform(*train)
    except AttributeError:
        warn("This Pipeline has no attribute 'fit_transform'. Using 'fit' instead.")
        pipeline.fit(*train)

    print(round(test.mae_percentage(pipeline), 4))


if __name__ == '__main__':
    main()
