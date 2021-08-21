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

#Load data
train_input = pd.read_csv("D:\\Lab\\Stock\\File\\Baseline_data\\Train\\현대차.csv",usecols=[1,2,3,4,5])
train_input = np.array(train_input)
train_output = pd.read_csv("D:\\Lab\\Stock\\File\\Baseline_data\\Train\\현대차.csv",usecols=[6])
train_output = np.array(train_output)
train_output.ravel()

val_input = pd.read_csv("D:\\Lab\\Stock\\File\\Baseline_data\\Val\\현대차.csv",usecols=[1,2,3,4,5])
val_input = np.array(val_input)
val_output = pd.read_csv("D:\\Lab\\Stock\\File\\Baseline_data\\Val\\현대차.csv",usecols=[6])
val_output = np.array(val_output)
val_output.ravel()

test_input = pd.read_csv("D:\\Lab\\Stock\\File\\Baseline_data\\Test\\현대차.csv",usecols=[1,2,3,4,5])
test_input = np.array(test_input)
test_output = pd.read_csv("D:\\Lab\\Stock\\File\\Baseline_data\\Test\\현대차.csv",usecols=[6])
test_output = np.array(test_output)
test_output.ravel()

from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

def XGBOOST_linear(g, regularization_value):
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=g, C=regularization_value))
    ])
    rbf_kernel_svm_clf.fit(train_input, train_output)

    y_pred_train = rbf_kernel_svm_clf.predict(train_input)
    train_acc = accuracy_score(train_output, y_pred_train)

    y_pred_val = rbf_kernel_svm_clf.predict(val_input)
    val_acc = accuracy_score(val_output, y_pred_val)

    y_pred_test = rbf_kernel_svm_clf.predict(test_input)
    test_acc = accuracy_score(test_output, y_pred_test)

    confusion = confusion_matrix(test_output, y_pred_test)
    precision = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    recall = confusion[0][0] / (confusion[0][0] + confusion[1][0])
    return train_acc, val_acc, test_acc, precision, recall

def svm_model_kernel_poly(d, regularization_value):
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=d, C=regularization_value))
    ])
    rbf_kernel_svm_clf.fit(train_input, train_output)

    y_pred_train = rbf_kernel_svm_clf.predict(train_input)
    train_acc = accuracy_score(train_output, y_pred_train)

    y_pred_val = rbf_kernel_svm_clf.predict(val_input)
    val_acc = accuracy_score(val_output, y_pred_val)

    y_pred_test = rbf_kernel_svm_clf.predict(test_input)
    test_acc = accuracy_score(test_output, y_pred_test)

    confusion = confusion_matrix(test_output, y_pred_test)
    precision = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    recall = confusion[0][0] / (confusion[0][0] + confusion[1][0])
    return train_acc, val_acc, test_acc, precision, recall

import numpy as np
_g = [i for i in np.arange(0.5,10.5,0.5)]
_d = [1,2,3,4]
_regularization_value_rbf = [0.5,1.5,10,100]
_regularization_value_poly = [0.5,1.5,10]
_chosen_kernel = ["rbf","poly"]
_train_acc, _val_acc, _test_acc, _precision, _recall = [], [],[], [],[]
with  open('SVM_Com_4.txt','w') as f:
    if _chosen_kernel[0]:
        f.write('[Gamma, C, train_acc, val_acc, test_acc, precision, recall]')
        f.write('\n')
        for g in _g:
            for c in _regularization_value_rbf:
                train_acc, val_acc, test_acc, precision, recall = XGBOOST_linear(g,c)
                _train_acc.append(train_acc)
                _val_acc.append(val_acc)
                _test_acc.append(test_acc)
                _precision.append(precision)
                _recall.append(recall)
                f.write('\n')
                f.write('[{}, {}, {}, {}, {}, {}, {}]'.format(g,c,train_acc,val_acc,test_acc,precision,recall))

    if _chosen_kernel[1]:
        f.write('\n')
        f.write('[Degree, C, train_acc, val_acc, test_acc, precision, recall]')
        for d in _d:
            for c in _regularization_value_poly:
                train_acc, val_acc, test_acc, precision, recall = svm_model_kernel_rbf(d,c)
                _train_acc.append(train_acc)
                _val_acc.append(val_acc)
                _test_acc.append(test_acc)
                _precision.append(precision)
                _recall.append(recall)
                f.write('\n')
                f.write('[{}, {}, {}, {}, {}, {}, {}]'.format(d,c,train_acc,val_acc,test_acc,precision,recall))

m = max(_val_acc)
print(m)
