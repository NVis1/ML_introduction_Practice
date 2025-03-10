import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lib import SklearnDataset


cache_path = "../cache"
memory_cache_path = str(cache_path + "/memory")


def target_tt_split(df: DataFrame, y_col_name: str,
                    train_size: float = 0.8,
                    test_size: float = 0.2,
                    random_state: int = 0
                    ):
    # df_print_details(df)

    y = df[[y_col_name]]
    X = df.drop(columns=[y_col_name], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state
    )

    train = SklearnDataset(X_train, y_train)
    valid = SklearnDataset(X_valid, y_valid)

    return train, valid


def preprocess(num_cols, cat_cols, num_pipeline_steps, cat_pipeline_steps):
    preprocessor = ColumnTransformer([
        ("num", Pipeline(num_pipeline_steps, memory=memory_cache_path), num_cols),
        ("cat", Pipeline(cat_pipeline_steps, memory=memory_cache_path), cat_cols)
    ])
    return preprocessor


def process():

    train, valid = target_tt_split(
        df=pd.read_csv("../datasets/HousingData.csv"),
        y_col_name="TAX"
    )

    num_cols = train.X.select_dtypes(include=["number"]).columns
    cat_cols = train.X.select_dtypes(include=["object"]).columns

    preprocessor = preprocess(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_pipeline_steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
        cat_pipeline_steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ],
    )

    pp = Pipeline([
        ("Preprocess", preprocessor),
        ("Model", LinearRegression(fit_intercept=True))
    ], memory=memory_cache_path)

    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
    new_column_names = num_cols + list(cat_feature_names)
    X_final = pd.DataFrame(X_transformed, columns=new_column_names)

    pp.fit(train.X, train.y)

    return mean_squared_error(valid.y, pp.predict(valid.X))


def main():
    res = process()
    print("\n", res)


if __name__ == '__main__':
    main()
