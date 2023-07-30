import functools
import itertools

from oswegonlp.constants import OFFSET
from oswegonlp import classifier_base, evaluation

import numpy as np
import math
from collections import defaultdict, Counter


# deliverable 3.1
def corpus_counts(x, y, label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """

    counts = defaultdict(int)
    for i in range(len(y)):
        if y[i] == label:
            for word in x[i]:
                val = 1 if counts[word] == 0 else counts[word] + 1
                counts.update({word: val})
    return counts


# deliverable 3.2
def estimate_pxy(x, y, label, alpha, vocab):
    """
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param alpha: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """

    probabilities = defaultdict(int)
    counts = corpus_counts(x, y, label)
    total_words_in_label = 0
    unique = list()
    for i in range(len(x)):
        if y[i] == label:
            for word in x[i]:
                unique.append(word)
    total_words_in_label = len(unique)
    for word in vocab:
        numerator = counts[word] + alpha
        denominator = total_words_in_label + (len(vocab) * alpha)
        prob = math.log(numerator / denominator)
        probabilities.update({word: prob})
    return probabilities

# deliverable 3.3
def estimate_nb(x, y, alpha):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """

    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)

    raise NotImplementedError


# deliverable 3.4
def find_best_smoother(x_tr, y_tr, x_dv, y_dv, alphas):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param alphas: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''

    raise NotImplementedError
