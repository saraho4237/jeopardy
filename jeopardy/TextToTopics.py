import argparse
import pickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class TextToTopics():

    def __init__(self,topic_model,vectorizer):
        self._vectorizer = vectorizer
        self._model = topic_model

    def fit(self, X, y):
        """
        Fit a latent topic model.
        """
        X = self._vectorizer.fit_transform(X)
        self._model.fit(X, y)
        return self

    def score(self, X, y):
        """Return a classification accuracy score on new data.

        Parameters
        ----------
        X: A numpy array or list of text fragments.
        y: A numpy array or python list of true class labels.
        """
        X = self._vectorizer.transform(X)
        return self._model.score(X, y)

def get_data(filename, text_column):
    """Load raw data from a json file and return text data from column of interest.

    Parameters
    ----------
    filename: The path to the json data file.
    text_column: The column to be vectorized and clustered (str).

    Returns
    -------
    X: A pd.Series containing the training text.
    """
    df=clean_data(filename)
    return (df[text_column])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit a Text Classifier model and save the results.')
    parser.add_argument('--data', help='A csv file with input data.')
    parser.add_argument('--out', help='A file to save the pickled model object to.')
    args = parser.parse_args()

    X = get_data(args.data)
    ttt_model = TextToTopics()
    ttt_model.fit(X)
    print(args.out)
    print(str(args.out))
    with open(args.out, 'w') as f:
        pickle.dump(ttt_model, f)
