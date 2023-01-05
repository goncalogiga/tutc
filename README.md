# Testing Unsupervised Text Classification (tutc) algorithms

This repository groups several methods that can be used to classify a corpus in an unsupervised manner. The three notebooks at the root of this directory demonstrate how each of these algorithms can be used to classify text without labels. Each method is simplified so the notebooks only call simple methods like ```fit()``` — all these wrappers around the algorithms used in the repository are written under ```/src```. 

## Brown clustering

The brown clustering core code is based on [yangyuan's brown-clustering git](https://github.com/yangyuan/brown-clustering/tree/master/brownclustering) and a simple ```BClustering``` wrapper class is added so it is compatible with sklearn ```BaseEstimator``` and ```TransformerMixin``` classes (helpfull for cross validation and other stuff)

Brown clustering groups similar words hierarchicaly together making it possible to classify texts using the clusters in wich their words are in.

## Zero shot classification

The zero-shot learning classifier is based on huggingface's zero-shot-classification pretrained model (facebook/bart-large-mnli). Zero (and few) shot learning is a recent technique used in NLP which consists in using a pretrained model that is able to generalize it's training labels to infer new labels, ie. the labels of the corpus we aim to classify in an unsupervised manner.

The notebook ```basic-zero-shot-classification``` also explores the use of brown clustering as a pretreatment method before applying zero-shot learning — the idea is to reduce the text only to it's most siginificant words. Though it reduces computation, removing context might harm the zero-shot classifier's performance.

## Simple match and Bert

The notebook ```simple_match_and_bert.ipynb``` explores a somewhat naive technique consisting in labeling the corpus using simple label matching (given a set of labels L, if text A as the word l in L in it, then it is labeled l) This results in a subset of the corpus that is now "labeled". Next, a powerfull transformer model like Bert is trained on this subset.

This method is not fitted to every kind of corpus since it the presence of labels is not always present in the corpus (for example, labels 'positive' and 'negative' will rarely appear in the training data) but might be a simple yet effective solution for some specific type of labels.