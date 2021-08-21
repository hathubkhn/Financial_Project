from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import pandas as pd
import pdb
import csv

train_input = pd.read_csv("D:\\Lab\\Stock\\File\\Study2\\KOSPI200\\Baseline_data\\Train\\현대차.csv",
                          usecols=[1, 2, 3, 4, 5])
train_input = np.array(train_input)
train_output = pd.read_csv("D:\\Lab\\Stock\\File\\Study2\\KOSPI200\\Baseline_data\\Train\\현대차.csv",
                           usecols=[6])
train_output = np.array(train_output)
train_output.ravel()

val_input = pd.read_csv("D:\\Lab\\Stock\\File\\Study2\\KOSPI200\\Baseline_data\\Val\\현대차.csv",
                        usecols=[1, 2, 3, 4, 5])
val_input = np.array(val_input)
val_output = pd.read_csv("D:\\Lab\\Stock\\File\\Study2\\KOSPI200\\Baseline_data\\Val\\현대차.csv",
                         usecols=[6])
val_output = np.array(val_output)
val_output.ravel()

test_input = pd.read_csv("D:\\Lab\\Stock\\File\\Study2\\KOSPI200\\Baseline_data\\Test\\현대차.csv",
                         usecols=[1, 2, 3, 4, 5])
test_input = np.array(test_input)
test_output = pd.read_csv("D:\\Lab\\Stock\\File\\Study2\\KOSPI200\\Baseline_data\\Test\\현대차.csv",
                          usecols=[6])
test_output = np.array(test_output)
test_output.ravel()

from sklearn.metrics import confusion_matrix

# Random Forest model
from sklearn.ensemble import RandomForestClassifier


def random_forest_model(n_trees, depth):
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, random_state=0)
    rf.fit(train_input, train_output)
    y_pred_train = rf.predict(train_input)
    train_acc = accuracy_score(train_output, y_pred_train)

    y_pred_val = rf.predict(val_input)
    val_acc = accuracy_score(val_output, y_pred_val)

    y_pred_test = rf.predict(test_input)
    test_acc = accuracy_score(test_output, y_pred_test)
    confusion = confusion_matrix(test_output, y_pred_test)
    precision = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    recall = confusion[0][0] / (confusion[0][0] + confusion[1][0])
    return train_acc, val_acc, test_acc, precision, recall


N_trees = list(range(10, 100, 10))
Max_depth = list(range(1, 4, 1))

_train_acc, _val_acc, _test_acc, _precision, _recall = [], [], [], [], []
with  open('D:\\Lab\\Financial project\\Baseline\\Results\\Study2\\KOSPI200\\Com4_RF.txt', 'w') as f:
    f.write('[N_trees, Max_depth, train_acc, val_acc, test_acc, precision, recall]')
    f.write('\n')
    for n_trees in N_trees:
        for max_depth in Max_depth:
            train_acc, val_acc, test_acc, precision, recall = random_forest_model(n_trees, max_depth)
            _train_acc.append(train_acc)
            _val_acc.append(val_acc)
            _test_acc.append(test_acc)
            _precision.append(precision)
            _recall.append(recall)
            f.write('\n')
            f.write('[{}, {}, {}, {}, {}, {}, {}]'.format(n_trees, max_depth, train_acc, val_acc, test_acc, precision,
                                                          recall))

m = max(_val_acc)
print(m)
