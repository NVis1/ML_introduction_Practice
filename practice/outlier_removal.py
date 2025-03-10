import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from lib import SklearnDataset


def get_mae(max_leaf_nodes, train: SklearnDataset, val: SklearnDataset):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train.X, train.y)
    preds_val = model.predict(val.X)
    mae = mean_absolute_error(val.y, preds_val)
    return mae


def remove_outlier_by_percentile(df, df_column):
    min_thresold, max_thresold = df_column.quantile( [0.001, 0.999] )
    return df[
        (df_column > min_thresold) & (df_column < max_thresold)
    ]


def remove_outlier_by_mad(df, df_column):
    return df[
        abs(df_column) < 4 * df_column.std()
    ]


def remove_outlier_by_zscore(df, df_column):
    return df[
        abs(
            (df_column - df_column.mean()) / df_column.std()
        ) < 4
    ]


def show(df, df_column):
    print(df_column.describe(), end="\n\n")
    print(df.shape, end="\n\n")
    sn.histplot(df_column, kde=True)
    plt.show()


def main():
    df = pd.read_csv("./bhp.csv")

    print(df.price_per_sqft.describe(), end="\n\n")
    print(df.shape, end="\n\n")

    df = remove_outlier_by_percentile(df, df.price_per_sqft)
    show(df, df.price_per_sqft)

    df = remove_outlier_by_zscore(df, df.price_per_sqft)
    show(df, df.price_per_sqft)

    df = remove_outlier_by_mad(df, df.price_per_sqft)
    show(df, df.price_per_sqft)


if __name__ == '__main__':
    main()
