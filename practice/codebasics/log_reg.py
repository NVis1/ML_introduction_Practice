import pandas as pd
from numpy import mean
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lib.funcs import mae_percentage


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


def main():
    df = pd.read_csv('../datasets/Titanic-Dataset.csv')

    # features = [""]
    target = "Survived"

    drop_features = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=drop_features, errors='ignore')

    y = df[target]
    X = df.drop(columns=target)

    preprocessor = make_standart_preprocessor_for(X)
    model = LogisticRegression(random_state=0)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ], memory=None)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=0)

    pipeline.fit(train_X, train_y)

    score = mean(cross_val_score(pipeline, test_X, test_y))
    print(score)
    score = mae_percentage(test_X, test_y, pipeline)
    print(score)


if __name__ == '__main__':
    main()
