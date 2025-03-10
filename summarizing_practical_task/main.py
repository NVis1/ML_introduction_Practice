import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from feature_engine.imputation import CategoricalImputer

from sklearn.pipeline import Pipeline

from lib import SklearnDataset, BetterPipeline

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier


def old_get_feature_importance_for(model, transformer):
    return pd.DataFrame({
        "Feature": transformer.get_feature_names_out(),
        "Importance": model.feature_importances_
    }).sort_values(by='Importance', ascending=False)


def main(df: pd.DataFrame):
    dataset = create_dataset(df, y_col_name="Churn")

    models = {
        "XGBoostRandomForest": XGBRFClassifier(
            n_estimators=42, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=42, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=42, random_state=42),
    }

    # ToDo Add GridSearchCV

    for name, model in models.items():
        print(f"\n\n==========MODEL TEST: {name}==========\n")
        handle_model(model, dataset)


def create_dataset(df, y_col_name: str):
    # print(df.dtypes)

    # Fixes error with TotalCharges column being incorrectly identified as non-categorical.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[y_col_name] = LabelEncoder().fit_transform(df[y_col_name])

    return SklearnDataset.from_df(
        df, y_col_name=y_col_name,
        remove_features=["customerID"],
        include_features=["InternetService", "OnlineSecurity", "Contract", "TechSupport",
                          "tenure", "MultipleLines", "PhoneService", "TotalCharges"]
    )


def handle_model(model, dataset):
    transformer = make_column_transformer(dataset.X)
    steps = {
        "transform": transformer,
        "model": model
    }

    train, test = dataset.target_tt_split()

    pl = BetterPipeline(steps=steps, memory=None)
    pl.fit_try_transform(*train)

    # res = dict(
    #     score=pl.score(*test),
    #     mae_score=pl.mae_percentage(*test),
    #     **metrics_test(test.y, pl.predict(test.X))
    # )
    res = pl.metrics(*test)

    print("Scores:")
    for k, v in res.items():
        print(f"\t{k}: {round(v, 4)}")

    print("\nFeature importance:")
    # importance_res = get_feature_importance_for(pl, *train)
    importance_res = pl.get_feature_importance_for(*train)
    for k, v in importance_res.items():
        print(f"\t{k}: {round(v, 4)}")


def make_column_transformer(X: pd.DataFrame):
    num_X = X.select_dtypes(include=["number", "int64", "float64"]).columns
    cat_X = X.select_dtypes(include=["object", "string"]).columns

    num_pl = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pl = Pipeline([
        ("imputer", CategoricalImputer()),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num_transformer", num_pl, num_X),
        ("cat_transformer", cat_pl, cat_X)
    ])


# implemented in BetterPipeline
def metrics_test(y_test, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
    )


# implemented in BetterPipeline
def get_feature_importance_for(model, X, y):
    importance = permutation_importance(
        model, X.sample(frac=0.2, random_state=42), y.sample(frac=0.2, random_state=42),
        scoring="accuracy", n_repeats=4, random_state=42, n_jobs=-1
    ).importances_mean

    return pd.Series(
        data=importance,
        index=X.columns
    ).sort_values(ascending=False)


if __name__ == '__main__':
    main(
        df=pd.read_csv("../datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    )
