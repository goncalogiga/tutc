try:
    from brownclustering.corpus import Corpus
except ModuleNotFoundError:
    raise Exception("""Please change the current directory in order to import
    brownclustering. Add the following line to the notebook before importation:

    %cd src/estimators/
    from bclustering_tutc import BCluserting
    %cd ../..
    """)

from brownclustering.core import BrownClustering
from sklearn.base import BaseEstimator, TransformerMixin


class BClustering(BaseEstimator, TransformerMixin):
    def __init__(self, m, alpha=1, start_symbol='<s>', end_symbol='</s>'):
        self.corpus = None
        self.m = m
        self.alpha = alpha
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol

    def fit(self, X, y=None):
        print("[I] Brown Clustering report:")
        self.corpus = Corpus(X, alpha=self.alpha,
                             start_symbol=self.start_symbol,
                             end_symbol=self.end_symbol)
        clustering = BrownClustering(self.corpus, self.m)
        clustering.train()
        return clustering

    def transform(self, X):
        return X
