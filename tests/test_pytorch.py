from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from oswegonlp import preprocessing, classifier_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logistic_regression
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

def setup_module():
    global x_tr, y_tr, x_dv, y_dv, counts_tr, x_dv_pruned, x_tr_pruned
    global labels
    global vocab
    global X_tr, X_tr_var, X_dv_var, Y_tr, Y_dv, Y_tr_var, Y_dv_var

    y_tr,x_tr = preprocessing.read_data('fakenews-train.csv',preprocessor=preprocessing.bag_of_words)
    labels = set(y_tr)

    counts_tr = preprocessing.aggregate_word_counts(x_tr)

    y_dv,x_dv = preprocessing.read_data('fakenews-dev.csv',preprocessor=preprocessing.bag_of_words)

    x_tr_pruned, vocab = preprocessing.prune_vocabulary(counts_tr, x_tr, 5)
    x_dv_pruned, _ = preprocessing.prune_vocabulary(counts_tr, x_dv, 5)

    ## remove this, so people can run earlier tests    
    X_tr = preprocessing.make_numpy(x_tr_pruned,vocab)
    X_dv = preprocessing.make_numpy(x_dv_pruned,vocab)
    label_set = sorted(list(set(y_tr)))
    Y_tr = np.array([label_set.index(y_i) for y_i in y_tr])
    Y_dv = np.array([label_set.index(y_i) for y_i in y_dv])

    X_tr_var = Variable(torch.from_numpy(X_tr.astype(np.float32)))
    X_dv_var = Variable(torch.from_numpy(X_dv.astype(np.float32)))

    Y_tr_var = Variable(torch.from_numpy(Y_tr))
    Y_dv_var = Variable(torch.from_numpy(Y_dv))

def test_d5_1_numpy():
    global x_dv, counts_tr
    
    x_dv_pruned, vocab = preprocessing.prune_vocabulary(counts_tr,x_dv,10)
    X_dv = preprocessing.make_numpy(x_dv_pruned,vocab)
    eq_(X_dv.sum(), 1925)
    eq_(X_dv.sum(axis=1)[4], 4)
    eq_(X_dv.sum(axis=1)[144], 5)

    eq_(X_dv.sum(axis=0)[10], 4)
    eq_(X_dv.sum(axis=0)[100], 2)

def test_d5_2_logreg():
    global X_tr, Y_tr, X_dv_var

    model = logistic_regression.build_linear(X_tr,Y_tr)
    scores = model.forward(X_dv_var)
    eq_(scores.size()[0], 328)
    eq_(scores.size()[1], 2)

def test_d5_3_log_softmax():

    scores = np.asarray([[-0.1721,-0.5167,-0.2574,0.1571],[-0.3643,0.0312,-0.4181,0.4564]], dtype=np.float32)
    ans = logistic_regression.log_softmax(scores)
    assert_almost_equals(ans[0][0], -1.3904355, places=5)
    assert_almost_equals(ans[1][1], -1.3458145, places=5)
    assert_almost_equals(ans[0][1], -1.7350391, places=5)

def test_d5_4_nll_loss():
    global X_tr, Y_tr, X_dv_var

    torch.manual_seed(765)
    model = logistic_regression.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
    loss = torch.nn.NLLLoss()
    logP = model.forward(X_tr_var)
    assert_almost_equals(logistic_regression.nll_loss(logP.data.numpy(), Y_tr), 0.6929905418230031, places=5)

def test_d5_5_accuracy():
    global Y_dv_var
    acc = evaluation.acc(np.load('logreg-es-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.8)

def test_d7_3_competition_dev1():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.835)

def test_d7_3_competition_dev2():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.84)

def test_d7_3_competition_dev3():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.85)

def test_d7_3_competition_dev4():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.86)

# todo: implement test for bakeoff rubric
