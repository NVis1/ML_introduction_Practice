from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from lib import BasicDataset


random_state = 42


def train(ds: BasicDataset, payloads, preprocessor):
    return search(
        searcher_kwargs=dict(
            cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
            scoring=make_scorer(f1_score, average='weighted'),
            n_jobs=5,
            refit=True,
        ),
        payloads=payloads,
        preprocessor=preprocessor,
        X=ds.X, y=ds.y
    )


def search(searcher_kwargs, payloads, preprocessor, X, y):
    results = []
    for payload in payloads:
        print(f"\tTesting model {payload["name"]}...")

        # pipeline = Pipeline(list({
        #     "preprocessor": preprocessor,
        #     "model": payload["estimator"]
        # }.items()))

        pipeline = Pipeline(
            preprocessor.steps + [("model", payload["estimator"])]
        )

        searcher = RandomizedSearchCV(
            random_state=random_state,
            estimator=pipeline,
            param_distributions=prefix_param_grid(payload["param_grid"], "model"),
            **searcher_kwargs
        )
        
        searcher.fit(X, y)

        results.append({
            "name": payload["name"],
            "searcher": searcher
        })
    return results


def prefix_param_grid(param_grid: dict, step_name: str) -> dict:
    return {f"{step_name}__{key}": value for key, value in param_grid.items()}

