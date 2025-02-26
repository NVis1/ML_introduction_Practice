import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("../datasets/canada_per_capita_income.csv")

    y = df[["per capita income (US$)"]]
    X = df.drop(["per capita income (US$)"], axis="columns")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    print(model.predict([[2020]]))


if __name__ == '__main__':
    main()
