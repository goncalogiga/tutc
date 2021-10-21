import yake
from yake.highlight import TextHighlighter

class Yake:
    def __init__(self, language = "en", max_ngram_size = 3, deduplication_thresold = 0.9,
                 deduplication_algo = 'seqm', windowSize = 1, numOfKeywords = 20):
        """
        @param text: A textual input for Yake. Can either be a string
        or a panda dataframe of multiple strings.
        """
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.deduplication_thresold = deduplication_thresold
        self.deduplication_algo = deduplication_algo
        self.windowSize = windowSize
        self.numOfKeywords = numOfKeywords

        self.kw_extractor = yake.KeywordExtractor(language, max_ngram_size,
                                                  deduplication_thresold,
                                                  deduplication_algo, windowSize,
                                                  numOfKeywords)

    def get_keywords(self, text, return_type="visual-highligh"):
        available_return_types = ["highligh", "visual-highligh", "dict", "list", "summary"]

        keywords = self.kw_extractor.extract_keywords(text)

        if return_type == available_return_types[0]:
            return TextHighlighter(self.max_ngram_size).highlight(text, keywords)
        elif return_type == available_return_types[1]:
            return TextHighlighter(self.max_ngram_size, highlight_pre="\033[1m",
                                   highlight_post="\033[0m").highlight(text, keywords)
        elif return_type == available_return_types[2]:
            return [kw for kw in keywords]
        elif return_type == available_return_types[3]:
            return [kw[0] for kw in keywords]
        elif return_type == available_return_types[4]:
            return " ".join([kw[0] for kw in keywords])
        else:
            raise Exception("No return_type \"%s\" was found." % return_type)
