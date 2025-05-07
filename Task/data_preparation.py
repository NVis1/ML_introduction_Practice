import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.impute import SimpleImputer

from lib import SklearnDataset


random_state = 42


def prepare_dataframe():
    df_original = pd.read_csv("./Train.csv").drop(["id"], axis="columns")
    df = df_original.drop(
        columns=["column_1", "column_2", "column_3", "column_4"]
    )

    df = clean(df, ["column_353", "column_312", "column_192", "column_129"])

    return df


def clean(df, drop_cols):
    return (df
            .drop(columns=drop_cols)
            .T.drop_duplicates().T
            .replace([np.inf, -np.inf], np.nan)
            )


def get_high_corr_columns(ds, threshold=0.95):
    corr_matrix = corr_matrix_for(ds.X)
    return [
        column for column in corr_matrix.columns
        if any(corr_matrix[column] >= threshold)
    ]


def get_not_high_corr_columns(ds, threshold=0.95):
    corr_matrix = corr_matrix_for(ds.X)
    return [
        column for column in corr_matrix.columns
        if any(corr_matrix[column] < threshold)
    ]


def handle_missing(ds):
    # return df.dropna()
    cols = ds.X.columns
    X = SimpleImputer(strategy="median").fit_transform(ds.X, ds.y)
    X = SimpleImputer(strategy="median", missing_values=np.inf).fit_transform(X, ds.y)
    ds.X = pd.DataFrame(X, columns=cols)
    return ds


def corr_matrix_for(X):
    print("\tBuilding correlation matrix...")
    corr_matrix = X.corr().abs()
    return corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape), k=1
        ).astype(bool)
    )


def balance(df):
    df_minor = df[df.column_5 >= 2]
    df_major = df[df.column_5 < 2].dropna()

    df_minor = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df_minor)
    )
    df_minor.columns = df.columns

    df_minorest = df_minor[df_minor.column_5 == 3]
    df_minor = pd.concat((df_minor, df_minorest, df_minorest, df_minorest),
                         axis="rows")

    return pd.concat((df_major, df_minor), axis="rows")


def treat_outliers(df):
    z_scores = np.abs( zscore(
        df.drop(columns=["column_5"]),
        nan_policy="omit"
    ))
    return df[ (z_scores < 4).all(axis=1) ]
