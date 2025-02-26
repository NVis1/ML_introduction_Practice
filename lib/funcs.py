from numpy import ndarray, mean
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from lib import Dataset


def make_standart_preprocessor_for(X: DataFrame):
    cat_X = X.select_dtypes(include=["object"]).columns
    num_X = X.select_dtypes(include=["number"]).columns

    num_transformer_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_transformer_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_transformer_pipeline, num_X),
        ("cat", cat_transformer_pipeline, cat_X)
    ])


"""

====================================================
==================== DEPRECATED ====================
====================================================

"""


# moved to Dataset class
def mae_percentage(X, y, pipeline) -> ndarray:
    return mean(-100 * cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring='neg_mean_absolute_error'
    ))


# moved to Dataset class as 2 funcs (from_df and target_tt_split).
# Same usage: Dataset.from_df(**kw).target_tt_split(**kw).
def target_tt_split(df: DataFrame, y_col_name: str,
                    remove_features: list[str] = None,
                    test_size: float = 0.2,
                    random_state: int = 0
                    ):
    """
    Returns train and valid Dataset objects
    """

    if remove_features is None:
        remove_features = []
    remove_features.append(y_col_name)

    y = df[y_col_name]
    X = df.drop(remove_features, axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    train = Dataset(X_train, y_train)
    valid = Dataset(X_valid, y_valid)

    return train, valid


def df_print_details(df):
    print(
        "\n=====Dataframe Info=====",
        df.info(),
        "\n=====Dataframe Head=====",
        df.head(),
        "\n=====Dataframe Description=====",
        df.describe(),
        "\n=====Dataframe Null Entries=====",
        df.isnull().sum(),

        sep="\n"
    )
