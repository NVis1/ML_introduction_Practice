from typing import Callable

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


class _DatasetBase:
    def __init__(self, X, y):
        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)

    def __iter__(self):
        return iter((self.X, self.y))


class Dataset(_DatasetBase):
    @classmethod
    def from_df(cls, df: pd.DataFrame, y_col_name: str, remove_features: list[str] = None):
        if remove_features is None:
            remove_features = []
        remove_features.append(y_col_name)

        y = df[y_col_name]
        X = df.drop(remove_features, axis=1)

        return cls(X, y)
    
    def mae_percentage(self, pipeline: Pipeline, cv=5) -> float:
        """
        Returns positive mean absolute error score for a given pipeline
        muptiplied by 100 (in percents) via cross-validation mean,
        calculated using dataset's X and y values.
        """
        return np.mean(-100 * cross_val_score(
            pipeline, self.X, self.y,
            cv=cv,
            scoring='neg_mean_absolute_error'
        ))

    def target_tt_split(self,
                        test_size: float = 0.2,
                        random_state: int = 42
                        ):
        """
        Returns train and valid Dataset objects
        """

        X_train, X_valid, y_train, y_valid = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state
        )

        train = Dataset(X_train, y_train)
        valid = Dataset(X_valid, y_valid)

        return train, valid
