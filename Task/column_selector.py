import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns=None):
        self.selected_columns = selected_columns

    def fit(self, X, y=None):
        if self.selected_columns is None:
            self.selected_columns = X.columns
        return self

    def transform(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_columns]
        else:
            return pd.DataFrame(X, columns=self.selected_columns)

    def get_feature_names_out(self):
        return self.selected_columns
