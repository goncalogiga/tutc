import yake
from sklearn.base import BaseEstimator, TransformerMixin
from yake.highlight import TextHighlighter
from tqdm import tqdm
# from tqdm.auto import tqdm  # for notebooks
tqdm.pandas()

class Yake(BaseEstimator, TransformerMixin):
    def __init__(self, language="en", max_ngram_size=3,
                 deduplication_thresold=0.9, deduplication_algo='seqm',
                 windowSize=1, numOfKeywords=20):

        self.language = language
        self.max_ngram_size = max_ngram_size
        self.deduplication_thresold = deduplication_thresold
        self.deduplication_algo = deduplication_algo
        self.windowSize = windowSize
        self.numOfKeywords = numOfKeywords

        self.kw_extractor = yake.KeywordExtractor(language, max_ngram_size,
                                                  deduplication_thresold,
                                                  deduplication_algo,
                                                  windowSize,
                                                  numOfKeywords)

    def get_keywords(self, text, result_type="visual-highligh"):
        available_return_types = ["highligh", "visual-highligh", "dict",
                                  "list", "summary"]

        kw = self.kw_extractor.extract_keywords(text)

        if result_type == available_return_types[0]:
            return TextHighlighter(self.max_ngram_size).highlight(text, kw)
        elif result_type == available_return_types[1]:
            return TextHighlighter(self.max_ngram_size,
                                   highlight_pre="\033[1m",
                                   highlight_post="\033[0m").highlight(text,
                                                                       kw)
        elif result_type == available_return_types[2]:
            return list(kw)
        elif result_type == available_return_types[3]:
            return [w[0] for w in kw]
        elif result_type == available_return_types[4]:
            return " ".join([w[0] for w in kw])
        else:
            raise Exception("No return_type \"%s\" was found." % result_type)

    def fit(self, X, y=None):
        return self

    def transform(self, X, result_type="list", quiet=False):
        if not quiet:
            print("[I] Transforming X using yake (returning type '%s')"
                  % result_type)
            return X.progress_apply(self.get_keywords, result_type=result_type)
        return X.apply(self.get_keywords, result_type=result_type)

    def fit_transform(self, X, y=None, result_type="list", quiet=False):
        return self.transform(X, result_type, quiet)
