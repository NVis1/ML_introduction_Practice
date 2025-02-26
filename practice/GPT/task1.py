import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.impute import SimpleImputer


def main():
    print("Loading dataset...")
    df = sns.load_dataset("titanic")

    print(df.info())
    print(df.head())
    print(df.describe())

    print("Null entries:\n", df.isnull().sum())

    # Separating table by datatypes for imputation
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns

    # Setuping imputers
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Using imputers on each column of their datatype and then replacing the original df data by them
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols]).astype(str)

    print("Null entries after imputation:\n", df.isnull().sum())

    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"], kde=True)
    plt.show(block=True)

    sns.countplot(data=df, x="survived")
    plt.show(block=True)


if __name__ == '__main__':
    main()
