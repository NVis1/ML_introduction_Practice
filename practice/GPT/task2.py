import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def main():
    print("Loading dataset...")
    df = sns.load_dataset("penguins")

    df_print_details(df)

    df = df.dropna()

    df_print_details(df)

    # Separating table by datatypes
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Setuping
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    df_num_cols = df[num_cols].copy()
    df_cat_cols = df[cat_cols].copy()

    # Using on each column of their datatype and then replacing the original df data by them
    df_num_cols = pd.DataFrame(
        scaler.fit_transform(df_num_cols)
    )
    df_cat_cols = pd.DataFrame(
        data=encoder.fit_transform(df_cat_cols),
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    # Concatenate numerical and encoded categorical data
    df = pd.concat([df_num_cols, df_cat_cols], axis=1)

    return df


if __name__ == '__main__':
    main()
