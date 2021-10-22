from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import torch as t
# !pip install transformers - for collab
import transformers
from transformers import pipeline


class ZeroShotClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, labels, *args, **kwargs):
        self.classifier = pipeline("zero-shot-classification")
        self.labels = labels

    def fit(self, X, y=None, quiet=False):
        classification_report = self.classifier(X, self.labels)

        if not quiet:
            target = classification_report[0]
            print("=== Classification Report ===")
            print("Sample 0:")
            print("\tInput: %s" % target["sequence"])
            print("\tLabels:")
            for label, score in zip(target["labels"], target["scores"]):
                print("\t\t %s - confidence at %f" % (label, score))

            if y is not None:
                correct_predictions = 0
                for report, true_label in zip(classification_report, y):
                    prediction = report["labels"][0]
                    if prediction == true_label:
                        correct_predictions += 1

                accuracy = correct_predictions/len(y)
                print("[I] accuracy:", accuracy)
                print("")

        return classification_report

    def transform(self, X):
        return X
