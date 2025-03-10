from typing import Callable
from warnings import warn

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


class BasicDataset(_DatasetBase):
    @classmethod
    def from_df(cls, df: pd.DataFrame,
                y_col_name: str,
                remove_features:  list[str] | None = None,
                include_features: list[str] | None = None):
        y = df[y_col_name]

        remove_features = remove_features or []
        remove_features.append(y_col_name)

        X = df.drop(remove_features, axis=1)
        if include_features:
            X = X[include_features]

        return cls(X, y)


class SklearnDataset(BasicDataset):
    """

    """

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

        train = SklearnDataset(X_train, y_train)
        valid = SklearnDataset(X_valid, y_valid)

        return train, valid

    """
    
    ====================================================
    ==================== DEPRECATED ====================
    ====================================================
    
    """

    # Moved to BetterPipeline.mae_percentage.
    def mae_percentage(self, pipeline: Pipeline, cv=5) -> float:
        """
        ========== DEPRECATED ==========
        Moved to BetterPipeline.mae_percentage. Consider using it instead.

        Returns positive mean absolute error score for a given pipeline muptiplied by 100 (in percents) via
        cross-validation mean, calculated using dataset's X and y values.
        """
        return np.mean(-100 * cross_val_score(
            pipeline, self.X, self.y,
            cv=cv,
            scoring='neg_mean_absolute_error'
        ))

    # ==================== DEPRECATED ====================
    # Moved to BetterPipeline.fit_try_transform.
    def train_and_transform_to(self, pipeline):
        """
        ========== DEPRECATED ==========
        Moved to BetterPipeline.fit_try_transform. Consider using it instead.
        """
        try:
            pipeline.fit_transform(self.X, self.y)
        except AttributeError:
            warn("This Pipeline has no attribute 'fit_transform'. Using 'fit' instead.")
            pipeline.fit(self.X, self.y)
