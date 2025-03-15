import pandas as pd
from feature_engine.imputation import CategoricalImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from lib import BetterPipeline


class ModelHandler:
    transformer = None
    train = None
    test = None
    grid_search = None

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def handle(self, estimator, param_grid):
        """
        - Обработать пропущенные значения, категориальные признаки.
        - Оценить качество модели (метрики: accuracy, precision, recall, F1-score).
        - Провести анализ значимости признаков.
        """

        grid_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=5,
            refit=True,
            random_state=42
        )

        pl = BetterPipeline(steps={
            "transform": self.transformer,
            "model": grid_search
        })
        pl.fit_try_transform(*self.train)

        print("\nBest parameters found:", grid_search.best_params_)

        print("\nScores:")
        metrics_res = pl.metrics(*self.test)
        for k, v in metrics_res.items():
            print(f"\t{k}: {round(v, 4)}")

        print("\nFeature importance:")
        importance_res = pl.get_feature_importance_for(*self.train)
        for k, v in importance_res.items():
            print(f"\t{k}: {round(v, 4)}")

        return pl

    def make_column_transformer(self, X: pd.DataFrame):
        num_X = X.select_dtypes(include=["number", "int64", "float64"]).columns
        cat_X = X.select_dtypes(include=["object", "string"]).columns

        num_pl = BetterPipeline({
            "imputer": SimpleImputer(strategy="median"),
            "scaler": StandardScaler(),
        })

        cat_pl = BetterPipeline({
            "imputer": CategoricalImputer(),
            "encoder": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        })

        self.transformer = ColumnTransformer([
            ("num_transformer", num_pl, num_X),
            ("cat_transformer", cat_pl, cat_X)
        ])
