from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


classic_pipeline = Pipeline(steps=[
    ("preprocessor", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(n_estimators=100, random_state=0))
])


def crossv_maes(X, y, pipeline) -> ndarray:
    return -1 * cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring='neg_mean_absolute_error'
    )


print("Average MAE score:", crossv_maes(None, None).mean())
