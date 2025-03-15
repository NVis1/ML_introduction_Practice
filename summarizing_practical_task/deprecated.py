import pandas as pd
from sklearn.inspection import permutation_importance


"""

====================================================
==================== DEPRECATED ====================
====================================================

"""


# implemented in BetterPipeline
def metrics_test(y_test, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
    )


# implemented in BetterPipeline
def get_metrics(pl, test):
    return dict(
        score=pl.score(*test),
        mae_score=pl.mae_percentage(*test),
        **metrics_test(test.y, pl.predict(test.X))
    )


# implemented in BetterPipeline
def get_feature_importance_for(model, X, y):
    importance = permutation_importance(
        model, X.sample(frac=0.2, random_state=42), y.sample(frac=0.2, random_state=42),
        scoring="accuracy", n_repeats=4, random_state=42, n_jobs=-1
    ).importances_mean

    return pd.Series(
        data=importance,
        index=X.columns
    ).sort_values(ascending=False)
