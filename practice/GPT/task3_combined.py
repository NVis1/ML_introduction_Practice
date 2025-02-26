from statistics import LinearRegression

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lib.funcs import df_print_details


def preprocess(X):
    num_X = X.select_dtypes(include=["number"]).columns
    cat_X = X.select_dtypes(include=["object"]).columns

    num_X_imputed = SimpleImputer(strategy="median").fit_transform(num_X)
    cat_X_imputed = SimpleImputer(strategy="most_frequent").fit_transform(cat_X)

    preprocessed_X_num = StandardScaler().fit_transform(num_X_imputed)
    preprocessed_X_cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(cat_X_imputed)

    return preprocessed_X_num, preprocessed_X_cat


def main():
    df = pd.read_csv("../datasets/HousingData.csv")
    df_print_details(df)

    y = df[["TAX"]]
    X = df.drop(["TAX"])

    preprocessed_X_num, preprocessed_X_cat = preprocess(X)

    X

    train_X, valid_X, train_y, valid_y = train_test_split()

    model = LinearRegression()


if __name__ == '__main__':
    main()
