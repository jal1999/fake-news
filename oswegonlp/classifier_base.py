from oswegonlp.constants import OFFSET
import numpy as np


# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]


# deliverable 2.1
def make_feature_vector(x, y):
    """
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param x: counter of base features
    :param y: label string
    :returns: dict of features, f(x,y)
    :rtype: dict
    """

    feat_vec = {}
    for word in x:
        feat_vec.update({(y, word): x[word]})
    feat_vec.update({(y, OFFSET): 1})
    return feat_vec


# deliverable 2.2
def predict(x, weights, labels):
    """
    prediction function

    :param x: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """

    scores = {'real': 0, 'fake': 0}
    real_feat_vec = make_feature_vector(x, 'real')
    fake_feat_vec = make_feature_vector(x, 'fake')
    for feat in real_feat_vec:
        val = scores[feat] + (weights[feat] * x[feat])
        scores.update({feat: val})
    for feat in fake_feat_vec:
        val = scores[feat] + (weights[feat] * x[feat])
        scores.update({feat: val})
    return argmax(scores), scores


def predict_all(x, weights, labels):
    """
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    """
    y_hat = np.array([predict(x_i, weights, labels)[0] for x_i in x])
    return y_hat
