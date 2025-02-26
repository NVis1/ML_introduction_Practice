from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, _BaseNB

from lib.Dataset_class import Dataset


def make_my_nb_pipeline(nb: _BaseNB):
    return Pipeline([
        # ("preprocessing", CountVectorizer()),
        ("training", nb),
    ], memory=None)


def main():
    dataset = Dataset( *datasets.load_wine(return_X_y=True, as_frame=True) )
    train, test = dataset.target_tt_split(random_state=1)

    gaus_pipeline = make_my_nb_pipeline(GaussianNB())
    mnnb_pipeline = make_my_nb_pipeline(MultinomialNB())

    gaus_pipeline.fit(*train)
    mnnb_pipeline.fit(*train)

    print(f"GaussianNB score: \t\t{test.mae_percentage(gaus_pipeline)};\n"
          f"MultinominalNB score: \t{test.mae_percentage(mnnb_pipeline)}")


if __name__ == '__main__':
    main()
